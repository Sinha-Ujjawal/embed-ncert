import os
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from textwrap import dedent
from typing import Any, Iterator, Sequence

import mlflow
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app_config import AppConfig

load_dotenv()

mlflow.langchain.autolog()

SQLITE3_DB = './data.db'


@dataclass(slots=True)
class AgentRequest:
    thread_id: str
    question: str
    conf: str
    k: int = 100  # Max no. of relevant docs to retrieve from Vector Store
    reranking_k: int = 5  # Max no. of relevant docs to retrieve while reranking
    min_words: int = 5


class Status(StrEnum):
    IN_PROGRESS = 'in_progress'
    DONE = 'done'


@dataclass(slots=True)
class AgentResponse:
    thread_id: str
    mlflow_run_id: str
    status: Status
    message: str
    data: dict[str, Any] = field(default_factory=lambda: {})


def ensure_tables():
    with sqlite3.connect(SQLITE3_DB) as conn:
        conn.execute('pragma journal_mode = WAL;')
        conn.execute("""create table if not exists agent_conversation_history (
            thread_id     text     not null,
            mlflow_run_id text     not null,
            timestamp_utc datetime not null,
            chunk_id      integer  not null,
            messages      blob     not null
        )""")


def save_messages_to_db(
    *,
    thread_id: str,
    mlflow_run_id: str,
    timestamp_utc: datetime,
    chunk_id: int,
    messages: Sequence[AnyMessage],
):
    with sqlite3.connect(SQLITE3_DB) as conn:
        conn.execute('pragma journal_mode = WAL;')
        conn.execute(
            """
            insert into agent_conversation_history
            (thread_id, mlflow_run_id, timestamp_utc, chunk_id, messages)
            values
            (?, ?, ?, ?, ?);
            """,
            (thread_id, mlflow_run_id, timestamp_utc, chunk_id, pickle.dumps(messages)),
        )


def fetch_history_from_fb(thread_id: str) -> Sequence[AnyMessage]:
    with sqlite3.connect(SQLITE3_DB) as conn:
        records = conn.execute(
            """
            select timestamp_utc, messages
            from agent_conversation_history
            where thread_id = ?
            order by timestamp_utc asc
            """,
            (thread_id,),
        ).fetchall()
    messages: Sequence[HumanMessage | AIMessageChunk] = [
        msg
        for _, blob in sorted(records, key=lambda record: record[0])
        for msg in pickle.loads(blob)
    ]
    compressed: Sequence[HumanMessage | AIMessage] = []
    current_running_ai_messages: Sequence[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            if current_running_ai_messages:
                compressed.append(AIMessage(content=''.join(current_running_ai_messages)))
                current_running_ai_messages.clear()
            compressed.append(message)
        elif isinstance(message, AIMessage):
            current_running_ai_messages.append(str(message.content))
    if current_running_ai_messages:
        compressed.append(AIMessage(content=''.join(current_running_ai_messages)))
        current_running_ai_messages.clear()
    return compressed  # type: ignore


@mlflow.trace
def retrieve_relevant_docs(request: AgentRequest) -> Sequence[Document]:
    app_config = AppConfig.from_yaml(request.conf)
    embeddings = app_config.embedding_config.langchain_embedding()
    vector_store = app_config.vector_store_config.get_vectorstore(embeddings)
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(input=request.question, k=request.k)
    docs = [doc for doc in docs if len(doc.page_content.split()) >= request.min_words]
    reranking_embeddings = app_config.reranking_embedding_config.langchain_embedding()
    reranking_retriever = FAISS.from_documents(docs, embedding=reranking_embeddings).as_retriever()
    docs = reranking_retriever.invoke(input=request.question, k=request.reranking_k)
    return docs


@mlflow.trace
def qa_using_llm(
    *,
    request: AgentRequest,
    history: Sequence[AnyMessage],
    docs: Sequence[Document],
    batch_size: int = 32,
) -> Iterator[Sequence[AIMessageChunk]]:
    rag_prompt_template = PromptTemplate(
        template=dedent("""
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.

            # IMPORTANT INSTRUCTIONS:
            1. Read through ALL the provided context chunks carefully
            2. Each chunk has a unique "Chunk ID" (e.g., chunk_001, chunk_002, etc.)
            3. Only use information from chunks that are directly relevant to answering the question
            4. In your final chunk_ids list, include ONLY the exact chunk IDs that contributed to your answer

            # Conversation History:
            <conversation_history>
            {conversation_history}
            </conversation_history>

            # Current Question:
            {question}

            # Context (with Chunk IDs):
            {context}

            # Respond in the format:
            Mention the relevent chunk ids in your response in format:
            [$chunk_id] content

            # CRITICAL
            - ONLY MENTION THE CHUNKS THAT IS RELEVANT TO THE QUESTION
            - MAINTAIN CONVERSATION CONTINUITY WHILE GROUNDING YOUR ANSWER IN THE NEW CONTEXT
            """),
        input_variables=['conversation_history', 'question', 'context'],
    )

    llm = ChatOllama(
        model=os.environ.get('RAG_LLM_MODEL', 'qwen3:4b'),
        reasoning=os.environ.get('RAG_LLM_THINKING', '0') == '1',
        base_url=os.environ.get('RAG_OLLMA_HOST'),
        temperature=0.0,
    )
    llm_chain = rag_prompt_template | llm
    conversation_history = '\n\n'.join(history.pretty_repr() for history in history)
    context = '\n\n'.join(
        f'## Chunk ID: {doc.metadata.get("chunk_id")}\n{doc.page_content}' for doc in docs
    )
    batch = []
    for chunk in llm_chain.stream(
        {
            'conversation_history': conversation_history,
            'question': request.question,
            'context': context,
        }
    ):
        batch.append(chunk)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch):
        yield batch


@mlflow.trace
def workflow(request: AgentRequest, mlflow_run_id: str) -> Iterator[AgentResponse]:
    def make_res(status: Status, message: str, **kwargs) -> AgentResponse:
        return AgentResponse(
            thread_id=request.thread_id,
            mlflow_run_id=mlflow_run_id,
            status=status,
            message=message,
            data=kwargs,
        )

    futs = []

    ensure_tables()

    yield make_res(Status.IN_PROGRESS, 'Retrieve history...')
    history = fetch_history_from_fb(request.thread_id)
    yield make_res(Status.DONE, 'History fetched', history=[msg.dict() for msg in history])

    with ThreadPoolExecutor() as executor:
        futs.append(
            executor.submit(
                save_messages_to_db,
                thread_id=request.thread_id,
                mlflow_run_id=mlflow_run_id,
                timestamp_utc=datetime.now(timezone.utc),
                chunk_id=0,
                messages=[HumanMessage(request.question)],
            )
        )

        yield make_res(Status.IN_PROGRESS, 'Retrieve relevant docs...')
        docs = retrieve_relevant_docs(request)
        yield make_res(Status.DONE, 'Relevant docs retrieved', docs=[doc.dict() for doc in docs])

        yield make_res(Status.IN_PROGRESS, 'Asking the question to LLM with relevant docs')
        llm_chunks: Sequence[AIMessageChunk]
        for batch_idx, llm_chunks in enumerate(
            qa_using_llm(request=request, history=history, docs=docs), 1
        ):
            yield make_res(
                Status.IN_PROGRESS,
                f'Generated batch: {batch_idx}',
                llm_chunks=[chunk.dict() for chunk in llm_chunks],
            )
            futs.append(
                executor.submit(
                    save_messages_to_db,
                    thread_id=request.thread_id,
                    mlflow_run_id=mlflow_run_id,
                    timestamp_utc=datetime.now(timezone.utc),
                    chunk_id=batch_idx,
                    messages=llm_chunks,
                )
            )
        yield make_res(Status.DONE, 'All chunks generated', workflow_completed=True)
        wait(futs)
        for fut in futs:
            fut.result()


def run_workflow(request: AgentRequest) -> Iterator[AgentResponse]:
    with mlflow.start_run(run_name=f'run_{request.thread_id}') as run:
        yield from workflow(request, mlflow_run_id=run.info.run_id)
