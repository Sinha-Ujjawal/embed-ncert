from abc import abstractmethod
from dataclasses import dataclass

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


@dataclass
class TokenizerConfig:
    @abstractmethod
    def docling_tokenizer(self) -> BaseTokenizer:
        raise NotImplementedError


@dataclass
class HuggingfaceTokenizerConfig(TokenizerConfig):
    model: str
    max_tokens: int | None = None

    def docling_tokenizer(self) -> BaseTokenizer:
        return HuggingFaceTokenizer.from_pretrained(
            model_name=self.model, max_tokens=self.max_tokens
        )
