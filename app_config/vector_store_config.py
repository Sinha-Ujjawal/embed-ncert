from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


@dataclass
class VectorStoreConfig:
    @abstractmethod
    def get_vectorstore(self, embeddings: Embeddings) -> VectorStore:
        raise NotImplementedError

    @abstractmethod
    def from_documents(self, docs: Iterable[Document], embeddings: Embeddings) -> VectorStore:
        raise NotImplementedError


@dataclass
class QdrantClientConfig:
    @abstractmethod
    def client(self) -> QdrantClient:
        raise NotImplementedError


@dataclass
class QdrantURLClientConfig(QdrantClientConfig):
    url: str

    def client(self) -> QdrantClient:
        return QdrantClient(url=self.url)


@dataclass
class QdrantInMemoryClientConfig(QdrantClientConfig):
    def client(self) -> QdrantClient:
        return QdrantClient(':memory:')


@dataclass
class QdrantOnDiskClientConfig(QdrantClientConfig):
    path: str

    def client(self) -> QdrantClient:
        return QdrantClient(path=self.path)


@dataclass
class QdrantVectorStoreConfig(VectorStoreConfig):
    client_config: QdrantClientConfig
    collection_name: str
    addnl_conf: dict[str, Any] = field(default_factory=lambda: {})

    def get_vectorstore(self, embeddings: Embeddings) -> VectorStore:
        return self.from_documents([], embeddings)

    def from_documents(self, docs: Iterable[Document], embeddings: Embeddings) -> VectorStore:
        return QdrantVectorStore.from_documents(
            client=self.client_config.client(),
            documents=list(docs),
            embedding=embeddings,
            collection_name=self.collection_name,
            **self.addnl_conf,
        )
