import argparse
import os
from textwrap import dedent
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app_config import AppConfig


class ResponseFormat(BaseModel):
    """Always use this tool to structure your response to the user."""

    thinking: str = Field(
        description='Your internal reasoning and thought process before arriving at the answer'
    )
    dont_know: bool = Field(
        description='Set this to True if you dont know the answer to the question'
    )
    answer: str = Field(description='Your answer to the question')
    chunk_ids: list[str] = Field(
        description='List of chunk IDs from the context that were used to answer the question'
    )


response_format_parser = PydanticOutputParser(pydantic_object=ResponseFormat)

RAG_PROMPT_TEMPLATE = PromptTemplate(
    template=dedent("""
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.

    If you don't know the answer, just say "I don't know" or "I'm not sure".
    Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}

    {format_instruction}
    """),
    input_variables=['question', 'context'],
    partial_variables={'format_instruction': response_format_parser.get_format_instructions()},
)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Script for creating a Chat Like Interface for Q/A the Vector DB'
    )
    parser.add_argument(
        '--conf',
        required=True,
        help='Config yaml file to use. See conf/application folder for examples',
    )
    parser.add_argument(
        '--k',
        type=int,
        default=4,
        required=False,
        help='Number of Documents to return in retrieve step',
    )
    parser.add_argument('--question', required=True, help='Your question')
    args = parser.parse_args()
    app_config = AppConfig.from_yaml(args.conf)
    embeddings = app_config.embedding_config.langchain_embedding()
    vector_store = app_config.vector_store_config.get_vectorstore(embeddings)

    llm = ChatOllama(
        model=os.environ.get('RAG_LLM_MODEL', 'qwen3:4b'),
        reasoning=os.environ.get('RAG_LLM_THINKING', '0') == '1',
        base_url=os.environ.get('RAG_OLLMA_HOST'),
    )
    llm_chain = RAG_PROMPT_TEMPLATE | llm | response_format_parser

    class State(TypedDict):
        question: str
        context: list[Document]
        answer: ResponseFormat

    def retrieve(state: State) -> dict[str, Any]:
        retrieved_docs = vector_store.similarity_search(state['question'], k=args.k)
        return {'context': retrieved_docs}

    def augment_and_generate(state: State) -> dict[str, Any]:
        context_with_ids = '\n\n'.join(
            f'Chunk ID: {doc.metadata.get("chunk_id")}\nContent: {doc.page_content}'
            for doc in state['context']
        )
        response = llm_chain.invoke({'question': state['question'], 'context': context_with_ids})
        return {'answer': response}

    builder = (
        StateGraph(State)
        .add_node(retrieve)
        .add_node(augment_and_generate)
        .add_edge(START, 'retrieve')
        .add_edge('retrieve', 'augment_and_generate')
        .add_edge('augment_and_generate', END)
    )
    graph = builder.compile()
    result = graph.invoke({'question': args.question})  # type: ignore
    print(f'Context: {result["context"]}\n\n')
    print(f'Answer: {result["answer"]}')


if __name__ == '__main__':
    main()
