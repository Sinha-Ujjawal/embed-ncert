import pytest
from utils.errors import suppress_errors, suppress_errors_async


def test_suppress_errors_simple():
    @suppress_errors(err_value=-1)
    def success_func(x: int, y: int) -> int:
        return x + y

    assert success_func(1, 2) == 3

    @suppress_errors(err_value=-1)
    def failure_func(x: int, y: int) -> int:
        raise NotImplementedError

    assert failure_func(1, 2) == -1


@pytest.mark.asyncio
async def test_suppress_errors_simple_async():
    @suppress_errors_async(err_value=-1)
    async def success_func(x: int, y: int) -> int:
        return x + y

    assert await success_func(1, 2) == 3

    @suppress_errors_async(err_value=-1)
    async def failure_func(x: int, y: int) -> int:
        raise NotImplementedError

    assert await failure_func(1, 2) == -1
