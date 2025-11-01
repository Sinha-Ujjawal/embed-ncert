import time

import pytest
from utils.throttle import throttle, throttle_async


def test_throttle_simple():
    @throttle(seconds=1)
    def fast_func():
        return 'Hello'

    start = time.time()
    result = fast_func()
    end = time.time()
    assert result == 'Hello'  # result should be "Hello"
    assert end - start >= 1  # time taken should be atleast 1 seconds
    assert end - start <= 1.1  # time taken should not be very large than 1 seconds


def test_throttle_slow():
    @throttle(seconds=1)
    def slow_func():
        time.sleep(2)  # sleeping for 5 seconds
        return 'Hello'

    start = time.time()
    result = slow_func()
    end = time.time()
    assert result == 'Hello'  # result should be "Hello"
    assert end - start >= 2  # time taken should be atleast 2 seconds
    assert end - start <= 2.1  # time taken should not be very large than 2 seconds


def test_throttle_with_exception():
    class CustomException(Exception):
        def __init__(self, value):
            Exception.__init__(self)
            self.value = value

    @throttle(seconds=1)
    def fail_func():
        raise CustomException(value='Hello')

    start = time.time()
    result = None
    try:
        fail_func()
    except CustomException as e:
        result = e.value
    end = time.time()
    assert result == 'Hello'  # result should be "Hello"
    assert end - start >= 1  # time taken should be atleast 1 seconds
    assert end - start <= 1.1  # time taken should not be very large than 1 seconds


@pytest.mark.asyncio
async def test_throttle_async_simple():
    @throttle_async(seconds=1)
    async def fast_func():
        return 'Hello'

    start = time.time()
    result = await fast_func()
    end = time.time()
    assert result == 'Hello'  # result should be "Hello"
    assert end - start >= 1  # time taken should be atleast 1 seconds
    assert end - start <= 1.1  # time taken should not be very large than 1 seconds


@pytest.mark.asyncio
async def test_throttle_async_slow():
    @throttle_async(seconds=1)
    async def slow_func():
        time.sleep(2)  # sleeping for 5 seconds
        return 'Hello'

    start = time.time()
    result = await slow_func()
    end = time.time()
    assert result == 'Hello'  # result should be "Hello"
    assert end - start >= 2  # time taken should be atleast 2 seconds
    assert end - start <= 2.1  # time taken should not be very large than 2 seconds


@pytest.mark.asyncio
async def test_throttle_async_with_exception():
    class CustomException(Exception):
        def __init__(self, value):
            Exception.__init__(self)
            self.value = value

    @throttle_async(seconds=1)
    async def fail_func():
        raise CustomException(value='Hello')

    start = time.time()
    result = None
    try:
        await fail_func()
    except CustomException as e:
        result = e.value
    end = time.time()
    assert result == 'Hello'  # result should be "Hello"
    assert end - start >= 1  # time taken should be atleast 1 seconds
    assert end - start <= 1.1  # time taken should not be very large than 1 seconds
