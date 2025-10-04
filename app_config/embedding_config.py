from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_ollama.embeddings import OllamaEmbeddings


@dataclass
class EmbeddingConfig:
    @abstractmethod
    def langchain_embedding(self) -> Embeddings:
        raise NotImplementedError


@dataclass
class OllamaEmbeddingConfig(EmbeddingConfig):
    model: str
    base_url: str
    addnl_conf: dict[str, Any] = field(default_factory=lambda: {})

    def langchain_embedding(self) -> Embeddings:
        return OllamaEmbeddings(model=self.model, base_url=self.base_url, **self.addnl_conf)
