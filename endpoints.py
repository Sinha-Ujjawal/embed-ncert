import json
import os
import uuid
from enum import StrEnum
from typing import Annotated, Any, Iterator, Sequence

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages.utils import AnyMessage
from pydantic import BaseModel

import agent
from db import fetch_history_from_db


def register_endpoints(app: FastAPI):
    def stream_objs(objs: Iterator[Any]) -> Iterator[str]:
        for obj in objs:
            yield json.dumps(obj)
            yield '\n'

    class QueryRequest(BaseModel):
        question: str
        subject: str
        thread_id: str | None = None

        def app_config_path(self) -> str:
            conf_path = f'./conf/app/{self.subject}.yaml'
            if not os.path.exists(conf_path):
                conf_path = './conf/app/default.yaml'
            assert os.path.exists(conf_path)
            return conf_path

    class QueryResponseType(StrEnum):
        REASONING = 'reasoning'
        CONTENT = 'content'

    class QueryResponse(BaseModel):
        thread_id: str
        mlflow_run_id: str
        model_repr: str
        tag: str
        response_type: QueryResponseType
        response: str

    @app.post('/query/')
    async def query(request: QueryRequest) -> StreamingResponse:
        thread_id = request.thread_id
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        agent_request = agent.AgentRequest(
            thread_id=thread_id,
            question=request.question,
            subject=request.subject,
            conf=request.app_config_path(),
        )

        def stream_generator() -> Iterator[QueryResponse]:
            for agent_response in agent.run_workflow(agent_request):
                ai_response = agent_response.ai_response
                if ai_response is not None:
                    additional_data = ai_response.message.additional_kwargs
                    if 'reasoning_content' in additional_data:
                        yield QueryResponse(
                            thread_id=agent_response.thread_id,
                            mlflow_run_id=agent_response.mlflow_run_id,
                            model_repr=ai_response.model_repr,
                            tag=ai_response.tag,
                            response_type=QueryResponseType.REASONING,
                            response=additional_data['reasoning_content'],
                        )
                    if ai_response.message.content:
                        yield QueryResponse(
                            thread_id=agent_response.thread_id,
                            mlflow_run_id=agent_response.mlflow_run_id,
                            model_repr=ai_response.model_repr,
                            tag=ai_response.tag,
                            response_type=QueryResponseType.CONTENT,
                            response=str(ai_response.message.content),
                        )

        stream = stream_objs(map(lambda x: x.model_dump(), stream_generator()))
        return StreamingResponse(stream, media_type='application/json')

    class HistoryResponse(BaseModel):
        thread_id: str
        messages: Sequence[AnyMessage]

    @app.get('/history/')
    async def history(
        thread_id: Annotated[
            str, Query(title='thread_id', description='`thread_id` of the conversation thread')
        ],
    ) -> HistoryResponse:
        return HistoryResponse(thread_id=thread_id, messages=fetch_history_from_db(thread_id))
