import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from enum import Enum
from functools import wraps
from typing import Any, Type, TypeVar

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    UNIFORM = 'uniform'
    EXPONENTIAL = 'exponential'


@dataclass(slots=True)
class RetryConfig:
    max_retries: int = 0
    base_delay: float = 0
    strategy: RetryStrategy = RetryStrategy.UNIFORM
    exponential_base: float = 2.0
    jitter: bool = False
    jitter_low: float = 0.5
    jitter_high: float = 1.5

    def get_delay(self, attempt: int) -> float:
        if self.strategy == RetryStrategy.UNIFORM:
            delay = self.base_delay
        else:
            delay = self.base_delay * (self.exponential_base**attempt)

        if self.jitter:
            delay *= random.uniform(self.jitter_low, self.jitter_high)

        return delay


T = TypeVar('T')


def retry(
    retry_config: RetryConfig, retry_exceptions: tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = replace(retry_config)
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    logger.error(f'Error occurred (attempt {attempt + 1}): {e}')
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        logger.info(f'Retrying in {delay:.2f}s...')
                        time.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def retry_async(
    retry_config: RetryConfig, retry_exceptions: tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = replace(retry_config)
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    logger.error(f'Error occurred (attempt {attempt + 1}): {e}')
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        logger.info(f'Retrying in {delay:.2f}s...')
                        await asyncio.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper

    return decorator
