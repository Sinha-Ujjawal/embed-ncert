import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from enum import Enum
from functools import wraps
from typing import Any, Type, TypeVar


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


F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])


def retry(
    retry_config: RetryConfig,
    logger: logging.Logger | None = None,
    retry_exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = replace(retry_config)
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if logger:
                        logger.error(f'Error occurred (attempt {attempt + 1}): {e}')
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        if logger:
                            logger.info(f'Retrying in {delay:.2f}s...')
                        time.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper  # type: ignore

    return decorator


def retry_async(
    retry_config: RetryConfig,
    logger: logging.Logger | None = None,
    retry_exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[AsyncF], AsyncF]:
    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = replace(retry_config)
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if logger:
                        logger.error(f'Error occurred (attempt {attempt + 1}): {e}')
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        if logger:
                            logger.info(f'Retrying in {delay:.2f}s...')
                        await asyncio.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper  # type: ignore

    return decorator
