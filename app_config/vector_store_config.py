from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import QdrantVectorStore


@dataclass
class VectorStoreConfig:
    @abstractmethod
    def get_vectorstore(self, embeddings: Embeddings) -> VectorStore:
        raise NotImplementedError

    @abstractmethod
    def from_documents(self, docs: Iterable[Document], embeddings: Embeddings) -> VectorStore:
        raise NotImplementedError


@dataclass
class QdrantVectorStoreConfig(VectorStoreConfig):
    collection_name: str
    client_opts: dict[str, Any] = field(default_factory=lambda: {})
    addnl_conf: dict[str, Any] = field(default_factory=lambda: {})

    def get_vectorstore(self, embeddings: Embeddings) -> VectorStore:
        return self.from_documents([], embeddings)

    def from_documents(self, docs: Iterable[Document], embeddings: Embeddings) -> VectorStore:
        return QdrantVectorStore.from_documents(
            documents=list(docs),
            embedding=embeddings,
            collection_name=self.collection_name,
            **self.client_opts,
            **self.addnl_conf,
        )
