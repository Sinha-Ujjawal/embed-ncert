from typing import Iterator, Sequence, TypeVar

T = TypeVar('T')


def batched(items: Iterator[T], batch_size: int) -> Iterator[Sequence[T]]:
    batch = []
    for item in items:
        if len(batch) >= batch_size:
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch
