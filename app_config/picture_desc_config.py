from dataclasses import dataclass, field
from typing import Any, Protocol

from docling.datamodel.pipeline_options import (AnyUrl,
                                                PictureDescriptionApiOptions,
                                                PictureDescriptionBaseOptions,
                                                PictureDescriptionVlmOptions)


class PictureDescriptionConfig(Protocol):
    def docling_picture_description_options(self) -> PictureDescriptionBaseOptions: ...


@dataclass(slots=True)
class PictureDescriptionApiConfig:
    url: str = "http://localhost:8000/v1/chat/completions"
    headers: dict[str, str] = field(default_factory=lambda: {})
    params: dict[str, Any] = field(default_factory=lambda: {})
    timeout: float = 300
    concurrency: int = 1
    prompt: str = "Describe this image in detail. Don't miss out any details."
    provenance: str = ""

    def docling_picture_description_options(self) -> PictureDescriptionApiOptions:
        return PictureDescriptionApiOptions(
            url=AnyUrl(self.url),
            headers=self.headers,
            params=self.params,
            timeout=self.timeout,
            concurrency=self.concurrency,
            prompt=self.prompt,
            provenance=self.provenance,
        )


@dataclass(slots=True)
class PictureDescriptionVlmConfig:
    repo_id: str
    prompt: str
    generation_config: dict[str, Any] = field(
        default_factory=lambda: dict(max_new_tokens=200, do_sample=False)
    )

    def docling_picture_description_options(self) -> PictureDescriptionVlmOptions:
        return PictureDescriptionVlmOptions(
            repo_id=self.repo_id,
            prompt=self.prompt,
            generation_config=self.generation_config,
        )
