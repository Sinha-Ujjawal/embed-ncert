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
        description='Your internal reasoning and thought process before arriving at the answer. Explicitly mention which chunk IDs you are considering and using.'
    )
    dont_know: bool = Field(
        description='Set this to True if you dont know the answer to the question'
    )
    answer: str = Field(
        description='Form the answer to the question using ONLY the context provided. Only use chunks that are directly relevant to answering the question.'
    )
    chunk_ids: list[str] = Field(
        description='CRITICAL: List the exact chunk IDs (e.g., "chunk_123", "chunk_456") from the context that you actually used to formulate your answer. Only include chunk IDs whose content directly contributed to your response. Sort by relevance to the question (most relevant first).'
    )


response_format_parser = PydanticOutputParser(pydantic_object=ResponseFormat)

RAG_PROMPT_TEMPLATE = PromptTemplate(
    template=dedent("""
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.

    IMPORTANT INSTRUCTIONS:
    1. Read through ALL the provided context chunks carefully
    2. Each chunk has a unique "Chunk ID" (e.g., chunk_001, chunk_002, etc.)
    3. Only use information from chunks that are directly relevant to answering the question
    4. In your thinking, explicitly state which chunk IDs you are considering and which ones you decide to use
    5. In your final chunk_ids list, include ONLY the exact chunk IDs that contributed to your answer
    6. If you don't know the answer based on the provided context, set dont_know to True

    Question: {question}

    Context (with Chunk IDs):
    {context}

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
        temperature=0.0,
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
