import logging
import os
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterator, Sequence

import mlflow
from dotenv import load_dotenv

# from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app_config import AppConfig
from db import fetch_history_from_db, save_messages_to_db
from utils.batch import batched
from utils.mlflow_utils import mlflow_trace

load_dotenv()

mlflow.langchain.autolog()

logger = logging.getLogger()

PROJECT_ROOT = Path(__file__).parent


@dataclass(slots=True)
class AgentRequest:
    thread_id: str
    question: str
    subject: str
    conf: str
    k: int = 5  # Max no. of relevant docs to retrieve from Vector Store
    # TODO: WHEN RERANKING IS ENABLED, THEN DONT FORGET TO SET THE ABOVE K TO A HIGHER NUMBER, e.g- 100?
    # reranking_k: int = 5  # Max no. of relevant docs to retrieve while reranking
    min_words: int = 5


@dataclass(slots=True)
class AIResponse:
    model_repr: str
    tag: str
    llm_chunks: Sequence[AIMessageChunk]


@dataclass(slots=True)
class AgentResponse:
    thread_id: str
    mlflow_run_id: str
    message: str
    ai_response: AIResponse | None = None
    data: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(slots=True, kw_only=True)
class Workflow:
    request: AgentRequest
    mlflow_run_id: str
    batch_size: int

    def make_res(
        self, *, message: str, ai_response: AIResponse | None = None, **kwargs
    ) -> AgentResponse:
        return AgentResponse(
            thread_id=self.request.thread_id,
            mlflow_run_id=self.mlflow_run_id,
            message=message,
            ai_response=ai_response,
            data=kwargs,
        )

    @mlflow_trace
    def retrieve_relevant_docs(self) -> Sequence[Document]:
        app_config = AppConfig.from_yaml(self.request.conf)
        embeddings = app_config.embedding_config.langchain_embedding()
        retriever = app_config.vector_store_config.get_retriever(embeddings)
        docs = retriever.invoke(input=self.request.question, k=self.request.k)
        docs = [doc for doc in docs if len(doc.page_content.split()) >= self.request.min_words]
        # TODO: MAKE RERANKER FAST
        # reranking_embeddings = app_config.reranking_embedding_config.langchain_embedding()
        # reranking_retriever = FAISS.from_documents(docs, embedding=reranking_embeddings).as_retriever()
        # docs = reranking_retriever.invoke(input=request.question, k=request.reranking_k)
        return docs

    @mlflow_trace
    def qa_using_llm(
        self, *, history: Sequence[AnyMessage], docs: Sequence[Document]
    ) -> Iterator[tuple[str, AIMessageChunk]]:
        rag_prompt_template = PromptTemplate(
            template=dedent((PROJECT_ROOT / './assets/prompts/agent.txt').read_text()),
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
        model_repr = repr(llm)
        for msg in llm_chain.stream(
            {
                'conversation_history': conversation_history,
                'question': self.request.question,
                'context': context,
            }
        ):
            if isinstance(msg, AIMessageChunk):
                yield model_repr, msg

    @mlflow_trace
    def run(self) -> Iterator[AgentResponse]:
        logger.info(f'Agent started with {self.request=}')
        futs = []
        yield self.make_res(message='Retrieve history...')
        logger.info('Retrieving history...')
        history = fetch_history_from_db(self.request.thread_id)
        yield self.make_res(message='History fetched', history=[msg.dict() for msg in history])
        logger.info('History fetched')

        with ThreadPoolExecutor() as executor:
            futs.append(
                executor.submit(
                    save_messages_to_db,
                    thread_id=self.request.thread_id,
                    mlflow_run_id=self.mlflow_run_id,
                    subject=self.request.subject,
                    timestamp_utc=datetime.now(timezone.utc),
                    chunk_id=0,
                    model_repr='human',
                    messages=[HumanMessage(self.request.question)],
                )
            )

            yield self.make_res(message='Retrieve relevant docs...')
            logger.info('Retrieve relevant docs...')
            docs = self.retrieve_relevant_docs()
            yield self.make_res(
                message='Relevant docs retrieved', docs=[doc.dict() for doc in docs]
            )
            logger.info('Relevant docs retrieved')

            yield self.make_res(message='Asking the question to LLM with relevant docs')
            logger.info('Asking LLM')
            llm_chunks: Sequence[AIMessageChunk]
            for batch_idx, batch in enumerate(
                batched(
                    self.qa_using_llm(history=history, docs=docs),
                    batch_size=self.batch_size,
                ),
                1,
            ):
                if not batch:
                    continue
                model_repr = batch[0][0]
                llm_chunks = list(map(lambda pair: pair[1], batch))
                yield self.make_res(
                    message=f'Generated batch: {batch_idx}',
                    ai_response=AIResponse(model_repr=model_repr, tag='qa', llm_chunks=llm_chunks),
                )
                logger.info(f'LLM Batch: {batch_idx} Response Sent')
                futs.append(
                    executor.submit(
                        save_messages_to_db,
                        thread_id=self.request.thread_id,
                        mlflow_run_id=self.mlflow_run_id,
                        subject=self.request.subject,
                        timestamp_utc=datetime.now(timezone.utc),
                        chunk_id=batch_idx,
                        model_repr=model_repr,
                        messages=llm_chunks,
                    )
                )
            yield self.make_res(message='All chunks generated', workflow_completed=True)
            logger.info('All chunks sent')
            wait(futs)
            for fut in futs:
                fut.result()


def run_workflow(request: AgentRequest) -> Iterator[AgentResponse]:
    load_dotenv(override=True)
    batch_size = int(os.environ.get('AI_RESPONSE_BATCH_SIZE', '5'))
    assert batch_size >= 1
    with mlflow.start_run(run_name=f'run_{request.thread_id}') as run:
        yield from Workflow(
            request=request,
            mlflow_run_id=run.info.run_id,
            batch_size=batch_size,
        ).run()
