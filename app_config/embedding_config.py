from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings


@dataclass
class EmbeddingConfig:
    @abstractmethod
    def langchain_embedding(self) -> Embeddings:
        raise NotImplementedError


@dataclass
class HuggingFaceEmbeddingsConfig(EmbeddingConfig):
    model_name: str
    """Model name to use."""

    cache_folder: str | None = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""

    model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass to the Sentence Transformer model, such as `device`,
    `prompts`, `default_prompt_name`, `revision`, `trust_remote_code`, or `token`.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer"""

    encode_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the documents of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""

    query_encode_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the query of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""

    multi_process: bool = False
    """Run encode() on multiple GPUs."""

    show_progress: bool = False
    """Whether to show a progress bar."""

    def langchain_embedding(self) -> Embeddings:
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            cache_folder=self.cache_folder,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
            query_encode_kwargs=self.query_encode_kwargs,
            multi_process=self.multi_process,
            show_progress=self.show_progress,
        )


@dataclass
class OllamaEmbeddingConfig(EmbeddingConfig):
    model: str
    base_url: str
    addnl_conf: dict[str, Any] = field(default_factory=lambda: {})

    def langchain_embedding(self) -> Embeddings:
        return OllamaEmbeddings(model=self.model, base_url=self.base_url, **self.addnl_conf)
