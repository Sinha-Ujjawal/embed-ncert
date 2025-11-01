import json
import os
import uuid
from enum import StrEnum
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import agent

app = FastAPI()


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
        conf=request.app_config_path(),
    )

    def stream_generator() -> Iterator[QueryResponse]:
        for agent_response in agent.run_workflow(agent_request):
            data = agent_response.data
            if 'llm_chunks' in data:
                for chunk in data['llm_chunks']:
                    additional_data = chunk['additional_kwargs']
                    if 'reasoning_content' in additional_data:
                        yield QueryResponse(
                            thread_id=agent_response.thread_id,
                            mlflow_run_id=agent_response.mlflow_run_id,
                            response_type=QueryResponseType.REASONING,
                            response=additional_data['reasoning_content'],
                        )
                    elif 'content' in chunk:
                        yield QueryResponse(
                            thread_id=agent_response.thread_id,
                            mlflow_run_id=agent_response.mlflow_run_id,
                            response_type=QueryResponseType.CONTENT,
                            response=chunk['content'],
                        )

    stream = stream_objs(map(lambda x: x.model_dump(), stream_generator()))
    return StreamingResponse(stream, media_type='application/json')
