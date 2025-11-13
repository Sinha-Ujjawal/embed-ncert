import logging
import os
from contextlib import asynccontextmanager

import ngrok
from dotenv import load_dotenv
from fastapi import FastAPI

from endpoints import register_endpoints

load_dotenv(override=True)

logger = logging.getLogger(__name__)

NGROK_AUTH_TOKEN = os.environ['NGROK_AUTH_TOKEN']
APPLICATION_PORT = 8000


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Setting up ngrok Endpoint')
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)  # type: ignore
    ngrok.forward(addr=APPLICATION_PORT)
    yield
    logger.info('Tearing Down ngrok Endpoint')
    ngrok.disconnect()


app = FastAPI(lifespan=lifespan)
register_endpoints(app)
