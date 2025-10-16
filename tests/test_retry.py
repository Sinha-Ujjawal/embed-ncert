import pytest
from utils.retry import RetryConfig, RetryStrategy, retry, retry_async


# Sync tests
def test_simple_retry():
    max_retries = 10
    retry_config = RetryConfig(max_retries=max_retries, base_delay=0.1, jitter=False)
    ctx = {'count': 0}

    @retry(retry_config)
    def _inner(ctx):
        ctx['count'] += 1
        assert 1 == 2, 'Deliberately raising Assertion Error'

    try:
        _inner(ctx)
    except AssertionError:
        pass

    assert ctx['count'] == max_retries + 1


def test_retry_success_after_failures():
    max_retries = 5
    retry_config = RetryConfig(max_retries=max_retries, base_delay=0.05, jitter=False)
    ctx = {'count': 0}

    @retry(retry_config)
    def _inner(ctx):
        ctx['count'] += 1
        if ctx['count'] < 3:
            raise ValueError('Not yet')
        return 'success'

    result = _inner(ctx)

    assert ctx['count'] == 3
    assert result == 'success'


def test_retry_exponential():
    max_retries = 3
    retry_config = RetryConfig(
        max_retries=max_retries,
        base_delay=0.1,
        strategy=RetryStrategy.EXPONENTIAL,
        exponential_base=2.0,
        jitter=False,
    )
    ctx = {'count': 0}

    @retry(retry_config)
    def _inner(ctx):
        ctx['count'] += 1
        raise RuntimeError('Always fail')

    try:
        _inner(ctx)
    except RuntimeError:
        pass

    assert ctx['count'] == max_retries + 1


def test_retry_specific_exceptions():
    retry_config = RetryConfig(max_retries=3, base_delay=0.05, jitter=False)
    ctx = {'count': 0}

    @retry(retry_config, retry_exceptions=(ValueError, ConnectionError))
    def _inner(ctx):
        ctx['count'] += 1
        if ctx['count'] == 1:
            raise ValueError('Retry this')
        raise TypeError('Do not retry this')

    try:
        _inner(ctx)
    except TypeError:
        pass

    assert ctx['count'] == 2


def test_retry_no_retries():
    retry_config = RetryConfig(max_retries=0, base_delay=0.1, jitter=False)
    ctx = {'count': 0}

    @retry(retry_config)
    def _inner(ctx):
        ctx['count'] += 1
        raise ValueError('Fail')

    try:
        _inner(ctx)
    except ValueError:
        pass

    assert ctx['count'] == 1


def test_retry_with_jitter():
    max_retries = 3
    retry_config = RetryConfig(
        max_retries=max_retries, base_delay=0.1, jitter=True, jitter_low=0.5, jitter_high=1.5
    )
    ctx = {'count': 0}

    @retry(retry_config)
    def _inner(ctx):
        ctx['count'] += 1
        raise ValueError('Always fail')

    try:
        _inner(ctx)
    except ValueError:
        pass

    assert ctx['count'] == max_retries + 1


# Async tests
@pytest.mark.asyncio
async def test_simple_retry_async():
    max_retries = 10
    retry_config = RetryConfig(max_retries=max_retries, base_delay=0.1, jitter=False)
    ctx = {'count': 0}

    @retry_async(retry_config)
    async def _inner(ctx):
        ctx['count'] += 1
        assert 1 == 2, 'Deliberately raising Assertion Error'

    try:
        await _inner(ctx)
    except AssertionError:
        pass

    assert ctx['count'] == max_retries + 1


@pytest.mark.asyncio
async def test_retry_async_success_after_failures():
    max_retries = 5
    retry_config = RetryConfig(max_retries=max_retries, base_delay=0.05, jitter=False)
    ctx = {'count': 0}

    @retry_async(retry_config)
    async def _inner(ctx):
        ctx['count'] += 1
        if ctx['count'] < 3:
            raise ValueError('Not yet')
        return 'success'

    result = await _inner(ctx)

    assert ctx['count'] == 3
    assert result == 'success'


@pytest.mark.asyncio
async def test_retry_async_exponential():
    max_retries = 3
    retry_config = RetryConfig(
        max_retries=max_retries,
        base_delay=0.1,
        strategy=RetryStrategy.EXPONENTIAL,
        exponential_base=2.0,
        jitter=False,
    )
    ctx = {'count': 0}

    @retry_async(retry_config)
    async def _inner(ctx):
        ctx['count'] += 1
        raise RuntimeError('Always fail')

    try:
        await _inner(ctx)
    except RuntimeError:
        pass

    assert ctx['count'] == max_retries + 1


@pytest.mark.asyncio
async def test_retry_async_specific_exceptions():
    retry_config = RetryConfig(max_retries=3, base_delay=0.05, jitter=False)
    ctx = {'count': 0}

    @retry_async(retry_config, retry_exceptions=(ValueError, ConnectionError))
    async def _inner(ctx):
        ctx['count'] += 1
        if ctx['count'] == 1:
            raise ValueError('Retry this')
        raise TypeError('Do not retry this')

    try:
        await _inner(ctx)
    except TypeError:
        pass

    assert ctx['count'] == 2


@pytest.mark.asyncio
async def test_retry_async_no_retries():
    retry_config = RetryConfig(max_retries=0, base_delay=0.1, jitter=False)
    ctx = {'count': 0}

    @retry_async(retry_config)
    async def _inner(ctx):
        ctx['count'] += 1
        raise ValueError('Fail')

    try:
        await _inner(ctx)
    except ValueError:
        pass

    assert ctx['count'] == 1


@pytest.mark.asyncio
async def test_retry_async_with_jitter():
    max_retries = 3
    retry_config = RetryConfig(
        max_retries=max_retries, base_delay=0.1, jitter=True, jitter_low=0.5, jitter_high=1.5
    )
    ctx = {'count': 0}

    @retry_async(retry_config)
    async def _inner(ctx):
        ctx['count'] += 1
        raise ValueError('Always fail')

    try:
        await _inner(ctx)
    except ValueError:
        pass

    assert ctx['count'] == max_retries + 1
