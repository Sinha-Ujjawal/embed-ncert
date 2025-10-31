import asyncio
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])


def throttle(wait_delta_time: timedelta) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = datetime.now()
            try:
                return func(*args, **kwargs)
            finally:
                end = datetime.now()
                if wait_delta_time > end - start:
                    remaining_seconds = (wait_delta_time - (end - start)).total_seconds()
                    logger.info(f'Waiting for {remaining_seconds} seconds')
                    time.sleep(remaining_seconds)

        return wrapper  # type: ignore

    return decorator


def throttle_async(wait_delta_time: timedelta) -> Callable[[AsyncF], AsyncF]:
    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = datetime.now()
            try:
                return await func(*args, **kwargs)
            finally:
                end = datetime.now()
                if wait_delta_time > end - start:
                    remaining_seconds = (wait_delta_time - (end - start)).total_seconds()
                    logger.info(f'Waiting for {remaining_seconds} seconds')
                    await asyncio.sleep(remaining_seconds)

        return wrapper  # type: ignore

    return decorator
