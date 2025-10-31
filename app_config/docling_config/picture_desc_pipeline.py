import logging
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import PaginatedPipelineOptions
from docling.models.picture_description_api_model import (
    PictureDescriptionApiModel,
    PictureDescriptionApiOptions,
)
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.models.picture_description_vlm_model import (
    PictureDescriptionVlmModel,
    PictureDescriptionVlmOptions,
)
from docling.pipeline.standard_pdf_pipeline import PaginatedPipeline
from PIL import Image
from pydantic.networks import AnyUrl
from utils.api_image_request import api_image_request
from utils.retry import RetryConfig, retry
from utils.throttle import throttle

logger = logging.getLogger(__name__)


@dataclass
class PictureDescAnalyser:
    @abstractmethod
    def model(
        self, *, artifacts_path: Path | str | None, accelerator_options: AcceleratorOptions
    ) -> PictureDescriptionBaseModel:
        raise NotImplementedError


@dataclass(slots=True)
class PictureDescriptionAnalyserOpenAIApi(PictureDescAnalyser):
    url: str = 'http://localhost:8000/v1/chat/completions'
    headers: dict[str, str] = field(default_factory=lambda: {})
    params: dict[str, Any] = field(default_factory=lambda: {})
    timeout: float = 300
    concurrency: int = 1
    prompt: str = "Describe this image in detail. Don't miss out any details."
    provenance: str = ''
    retry_config: RetryConfig | None = None
    min_time_per_request_in_seconds: int | None = None  # For throttling of api requests

    def model(
        self, *, artifacts_path: Path | str | None, accelerator_options: AcceleratorOptions
    ) -> PictureDescriptionApiModel:
        model = PictureDescriptionApiModel(
            enabled=True,
            enable_remote_services=True,
            artifacts_path=artifacts_path,
            options=PictureDescriptionApiOptions(
                url=AnyUrl(self.url),
                headers=self.headers,
                params=self.params,
                timeout=self.timeout,
                concurrency=self.concurrency,
                prompt=self.prompt,
                provenance=self.provenance,
            ),
            accelerator_options=accelerator_options,
        )

        # patch _annotate_images for implementing retry logic
        def _patched_annotate_images(images: Iterable[Image.Image]) -> Iterable[str]:
            # Note: technically we could make a batch request here,
            # but not all APIs will allow for it. For example, vllm won't allow more than 1.

            def _api_request(image):
                return api_image_request(
                    image=image,
                    prompt=model.options.prompt,
                    url=model.options.url,
                    timeout=model.options.timeout,
                    headers=model.options.headers,
                    **model.options.params,
                )

            if self.min_time_per_request_in_seconds:
                _api_request = throttle(timedelta(seconds=self.min_time_per_request_in_seconds))(
                    _api_request
                )

            if self.retry_config:
                _api_request = retry(self.retry_config)(_api_request)

            with ThreadPoolExecutor(max_workers=model.concurrency) as executor:
                yield from executor.map(_api_request, images)

        model._annotate_images = _patched_annotate_images

        return model


@dataclass(slots=True)
class PictureDescriptionAnalyserVlm(PictureDescAnalyser):
    repo_id: str
    prompt: str
    generation_config: dict[str, Any] = field(
        default_factory=lambda: dict(max_new_tokens=200, do_sample=False)
    )

    def model(
        self, *, artifacts_path: Path | str | None, accelerator_options: AcceleratorOptions
    ) -> PictureDescriptionVlmModel:
        return PictureDescriptionVlmModel(
            enabled=True,
            enable_remote_services=True,
            artifacts_path=artifacts_path,
            options=PictureDescriptionVlmOptions(
                repo_id=self.repo_id,
                prompt=self.prompt,
                generation_config=self.generation_config,
            ),
            accelerator_options=accelerator_options,
        )


def picture_desc_analyser_pipeline_cls(
    paginated_pipeline_cls: type[PaginatedPipeline],
    picture_desc_analyser: PictureDescAnalyser,
) -> type:
    class PictureDescAnalyserPipeline(paginated_pipeline_cls):
        def __init__(self, pipeline_options: PaginatedPipelineOptions):
            super().__init__(pipeline_options)
            self.enrichment_pipe = (self.enrichment_pipe or []) + [
                picture_desc_analyser.model(
                    artifacts_path=pipeline_options.artifacts_path,
                    accelerator_options=pipeline_options.accelerator_options,
                )
            ]

            self.keep_backend = True

    return PictureDescAnalyserPipeline
