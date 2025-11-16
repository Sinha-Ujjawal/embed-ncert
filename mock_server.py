import time
from enum import StrEnum
from functools import wraps
from typing import Any, Callable, Iterator, ParamSpec

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# -- Type Variables ---
P = ParamSpec('P')


# --- Common Models ---
class DataMessage(BaseModel):
    data: Any


class ErrorMessage(BaseModel):
    error: str
    details: str | None = None


class ResponseStruct(BaseModel):
    status: int
    message: DataMessage | ErrorMessage

    @property
    def to_json_response(self) -> JSONResponse:
        return JSONResponse(content=self.model_dump(), status_code=self.status)


# --- Utils and Exception Handling ---
def stream_from_base_model_generator(
    generator_fn: Callable[P, Iterator[BaseModel]],
) -> Callable[P, StreamingResponse]:
    @wraps(generator_fn)
    def _inner(*args, **kwargs) -> StreamingResponse:
        def _iter():
            try:
                for obj in generator_fn(*args, **kwargs):
                    yield ResponseStruct(
                        message=DataMessage(data=obj), status=200
                    ).model_dump_json()
                    yield '\n'
            except Exception as exc:
                yield ResponseStruct(
                    message=ErrorMessage(error=str(exc)), status=500
                ).model_dump_json()
                raise

        return StreamingResponse(content=_iter())

    return _inner


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException) -> JSONResponse:
    return ResponseStruct(
        message=ErrorMessage(error=str(exc), details=str(exc.detail)),
        status=exc.status_code,
    ).to_json_response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError) -> JSONResponse:
    return ResponseStruct(message=ErrorMessage(error=str(exc)), status=400).to_json_response


# --- Endpoints ---
class ConvoRequest(BaseModel):
    question: str
    subject: str
    thread_id: str | None = None


class ConvoResponseType(StrEnum):
    REASONING = 'reasoning'
    CONTENT = 'content'


class ConvoResponse(BaseModel):
    thread_id: str
    mlflow_run_id: str
    model_repr: str
    tag: str
    response_type: ConvoResponseType
    response: str


@app.post('/convo/', response_model=None)
@stream_from_base_model_generator
def convo(request: ConvoRequest) -> Iterator[ConvoResponse]:
    LOREM_IPSUM = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."

    def mk_convo_res(res: str, res_type: ConvoResponseType) -> ConvoResponse:
        return ConvoResponse(
            thread_id=request.thread_id or 'dummy',
            mlflow_run_id='dummy',
            model_repr='dummy',
            tag='dummy',
            response=res,
            response_type=res_type,
        )

    for word in LOREM_IPSUM.split():
        time.sleep(0.5)
        yield mk_convo_res(word, ConvoResponseType.REASONING)
    for word in LOREM_IPSUM.split():
        time.sleep(0.5)
        yield mk_convo_res(word, ConvoResponseType.CONTENT)
