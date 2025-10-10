import asyncio
import base64
import logging
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import aiohttp
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import PaginatedPipelineOptions
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import PaginatedPipeline
from docling_core.types.doc import DoclingDocument, NodeItem, TextItem
from docling_core.types.doc.labels import DocItemLabel
from PIL.Image import Image

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

        async def _submit_and_wait() -> list[NodeItem]:
            sem = asyncio.Semaphore(self.concurrency)

            async def _submit(idx: int, el: ItemAndImageEnrichmentElement):
                assert isinstance(el.item, TextItem)
                async with sem:
                    text = await self.extract_text_from_img(el.image)
                return idx, text

            res = []
            tasks = []
            for idx, el in enumerate(element_batch):
                res.append(el.item)
                tasks.append(_submit(idx, el))
            for task in asyncio.as_completed(tasks):
                idx, text = await task
                res[idx].text = text

            return res

        yield from loop.run_until_complete(_submit_and_wait())


@dataclass(slots=True)
class TextEnhancerAnalyserOpenAIApi(TextEnhancerAnalyserAsync):
    """Configuration for OpenAI-compatible API client."""

    url: str = 'http://localhost:8000/v1/chat/completions'
    headers: dict[str, str] = field(default_factory=lambda: {})
    params: dict[str, Any] = field(default_factory=lambda: {})
    timeout: float = 60 * 1  # 1 minute
    prompt: str = 'Extract text from the image.'

    def encode_image_to_base64(self, image: Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    async def extract_text_from_img(self, image: Image) -> str:
        """Send image to OpenAI-compatible API and extract text."""
        try:
            # Convert image to base64
            base64_image = self.encode_image_to_base64(image)

            # Prepare the request payload
            payload = {
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': self.prompt},
                            {
                                'type': 'image_url',
                                'image_url': {'url': f'data:image/png;base64,{base64_image}'},
                            },
                        ],
                    }
                ],
                **self.params,  # Add any additional params like model, max_tokens, etc.
            }

            # Make the async HTTP request
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info(f'POST request to url: {self.url}')
                async with session.post(self.url, json=payload, headers=self.headers) as response:
                    response.raise_for_status()
                    result = await response.json()

                    # Extract the content from the response
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        logger.info(f'text extraction successful: {content}...')
                        return content
                    else:
                        logger.error(f'Unexpected response format: {result}')
                        return 'Error: Unexpected response format'

        except aiohttp.ClientError as e:
            logger.error(f'HTTP error calling API: {e}')
            return ''
        except Exception as e:
            logger.error(f'Error calling API: {e}')
            return ''


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
