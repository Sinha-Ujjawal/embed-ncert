import asyncio
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def throttle(
    *,
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
    milliseconds: int = 0,
    minutes: int = 0,
    hours: int = 0,
    weeks: int = 0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    wait_delta_time = timedelta(
        days=days,
        seconds=seconds,
        microseconds=microseconds,
        minutes=minutes,
        hours=hours,
        weeks=weeks,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
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

        return wrapper

    return decorator


def throttle_async(
    *,
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
    milliseconds: int = 0,
    minutes: int = 0,
    hours: int = 0,
    weeks: int = 0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    wait_delta_time = timedelta(
        days=days,
        seconds=seconds,
        microseconds=microseconds,
        minutes=minutes,
        hours=hours,
        weeks=weeks,
    )

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
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

        return wrapper

    return decorator
