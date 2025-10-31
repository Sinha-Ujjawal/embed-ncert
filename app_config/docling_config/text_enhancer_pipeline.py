import asyncio
import logging
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import PaginatedPipelineOptions
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import PaginatedPipeline
from docling_core.types.doc import DoclingDocument, NodeItem, TextItem
from docling_core.types.doc.labels import DocItemLabel
from PIL.Image import Image
from pydantic.networks import AnyUrl
from utils.api_image_request import api_image_request
from utils.retry import RetryConfig, retry_async
from utils.throttle import throttle_async

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TextEnhancerAnalyser:
    allow_label_expr: str = 'True'

    @abstractmethod
    def analyse(
        self, doc: DoclingDocument, element_batch: Iterable[ItemAndImageEnrichmentElement]
    ) -> Iterable[NodeItem]:
        raise NotImplementedError


@dataclass(slots=True)
class TextEnhancerAnalyserAsync(TextEnhancerAnalyser):
    concurrency: int = 1
    retry_config: RetryConfig | None = None
    min_time_per_request_in_seconds: int | None = None  # For throttling

    @abstractmethod
    async def extract_text_from_img(self, image: Image) -> str:
        raise NotImplementedError

    def analyse(
        self, doc: DoclingDocument, element_batch: Iterable[ItemAndImageEnrichmentElement]
    ) -> Iterable[NodeItem]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Patch extract_formula_from_img with retry if config is defined
        extract_func = self.extract_text_from_img
        if self.min_time_per_request_in_seconds is not None:
            extract_func = throttle_async(timedelta(seconds=self.min_time_per_request_in_seconds))(
                extract_func
            )
        if self.retry_config is not None:
            extract_func = retry_async(self.retry_config)(extract_func)

        async def _submit_and_wait() -> list[NodeItem]:
            sem = asyncio.Semaphore(self.concurrency)

            async def _submit(idx: int, el: ItemAndImageEnrichmentElement):
                assert isinstance(el.item, TextItem)
                async with sem:
                    formula_text = await extract_func(el.image)
                return idx, formula_text

            res = []
            tasks = []
            for idx, el in enumerate(element_batch):
                res.append(el.item)
                tasks.append(_submit(idx, el))
            for task in asyncio.as_completed(tasks):
                idx, formula_text = await task
                if formula_text:
                    res[idx].text = formula_text

            return res

        yield from loop.run_until_complete(_submit_and_wait())


@dataclass(slots=True)
class TextEnhancerAnalyserOpenAIApi(TextEnhancerAnalyserAsync):
    """Configuration for OpenAI-compatible API client."""

    url: AnyUrl = AnyUrl('http://localhost:8000/v1/chat/completions')
    headers: dict[str, str] = field(default_factory=lambda: {})
    params: dict[str, Any] = field(default_factory=lambda: {})
    timeout: float = 60 * 1  # 1 minute
    prompt: str = 'Extract text from the image.'

    async def extract_text_from_img(self, image: Image) -> str:
        """Send image to OpenAI-compatible API and extract text."""
        return api_image_request(
            image=image,
            prompt=self.prompt,
            url=self.url,
            timeout=self.timeout,
            header=self.headers,
            **self.params,
        )


class TextEnhancerAnalyserEnrichmentModel(BaseItemAndImageEnrichmentModel):
    images_scale = 2.6

    def __init__(self, text_analyser: TextEnhancerAnalyser):
        self.text_analyser = text_analyser

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return isinstance(element, TextItem) and eval(
            self.text_analyser.allow_label_expr,
            {'DocItemLabel': DocItemLabel, 'label': element.label},
        )

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        yield from self.text_analyser.analyse(doc, element_batch)


def text_enhancer_analyser_pipeline_cls(
    paginated_pipeline_cls: type[PaginatedPipeline],
    text_analyser: TextEnhancerAnalyser,
) -> type:
    class TextEnhancerAnalyserPipeline(paginated_pipeline_cls):
        def __init__(self, pipeline_options: PaginatedPipelineOptions):
            super().__init__(pipeline_options)
            self.enrichment_pipe = (self.enrichment_pipe or []) + [
                TextEnhancerAnalyserEnrichmentModel(text_analyser=text_analyser)
            ]

            self.keep_backend = True

    return TextEnhancerAnalyserPipeline
