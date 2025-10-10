import asyncio
import base64
import logging
import re
import subprocess
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import aiohttp
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import PaginatedPipelineOptions
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import PaginatedPipeline
from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem, TextItem
from PIL.Image import Image

logger = logging.getLogger(__name__)


def deduplicate_latex(latex_string, debug=False):
    """
    Deduplicate formulas in LaTeX arrays by removing duplicate cells.

    Args:
        latex_string (str): Input LaTeX string
        debug (bool): Print debug information

    Returns:
        str: Deduplicated LaTeX string
    """
    return deduplicate_latex_regex(latex_string, debug)


def normalize_cell(cell):
    """Normalize cell content for comparison by removing extra braces."""
    normalized = cell.strip()

    # Remove all outer brace pairs iteratively
    changed = True
    iterations = 0
    max_iterations = 10  # Safety limit

    while changed and iterations < max_iterations:
        iterations += 1
        changed = False

        # Try to remove double braces {{...}}
        if normalized.startswith('{{') and normalized.endswith('}}'):
            # Check if they match
            test = normalized[2:-2]
            if test:  # Don't reduce to empty string
                normalized = test
                changed = True
                continue

        # Try to remove single braces {...}
        if normalized.startswith('{') and normalized.endswith('}'):
            # Count braces to see if outer ones match
            inner = normalized[1:-1]
            if inner:  # Don't reduce to empty string
                depth = 0
                valid = True
                for char in inner:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                    if depth < 0:
                        valid = False
                        break

                if valid and depth == 0:
                    normalized = inner
                    changed = True

    return normalized.strip()


def deduplicate_latex_regex(latex_string, debug=False):
    """Regex-based deduplication for arrays."""

    def process_array(match):
        header = match.group(1)  # e.g., {l l}
        content = match.group(2)

        if debug:
            logger.debug('\nProcessing array with regex')
            logger.debug(f'Header: {header}')
            logger.debug(f'Content: {repr(content)}')

        rows = re.split(r'\\\\', content)
        seen_cells = set()
        new_rows = []

        for row_num, row in enumerate(rows):
            row = row.strip()
            if not row:
                continue

            cells = row.split('&')
            new_cells = []

            if debug:
                logger.debug(f'\nRow {row_num}: {len(cells)} cells')

            for cell in cells:
                cell_cleaned = cell.strip()
                if not cell_cleaned:
                    continue

                cell_normalized = normalize_cell(cell_cleaned)

                if debug:
                    logger.debug(f'  Cell: {repr(cell_cleaned)}')
                    logger.debug(f'  Normalized: {repr(cell_normalized)}')
                    logger.debug(f'  In seen? {cell_normalized in seen_cells}')

                if cell_normalized not in seen_cells:
                    seen_cells.add(cell_normalized)
                    new_cells.append(cell_cleaned)
                elif debug:
                    logger.debug('    -> REMOVED (duplicate)')

            if new_cells:
                new_rows.append('&'.join(new_cells))

        result = r'\begin{array}' + header + '\\\\'.join(new_rows) + r'\end{array}'

        if debug:
            logger.debug(f'\nResult: {result}')

        return result

    # Match array environments
    pattern = r'\\begin\{array\}(\{[^}]+\})(.*?)\\end\{array\}'
    result = re.sub(pattern, process_array, latex_string, flags=re.DOTALL)

    return result


@dataclass(slots=True)
class FormulaUnderstandingAnalyser:
    @abstractmethod
    def analyse(
        self, doc: DoclingDocument, element_batch: Iterable[ItemAndImageEnrichmentElement]
    ) -> Iterable[NodeItem]:
        raise NotImplementedError


@dataclass(slots=True)
class FormulaUnderstandingAnalyserAsync(FormulaUnderstandingAnalyser):
    concurrency: int = 1

    @abstractmethod
    async def extract_formula_from_img(self, image: Image) -> str:
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
                    formula_text = await self.extract_formula_from_img(el.image)
                return idx, formula_text

            res = []
            tasks = []
            for idx, el in enumerate(element_batch):
                res.append(el.item)
                tasks.append(_submit(idx, el))
            for task in asyncio.as_completed(tasks):
                idx, formula_text = await task
                res[idx].text = formula_text

            return res

        yield from loop.run_until_complete(_submit_and_wait())


@dataclass(slots=True)
class FormulaUnderstandingAnalyserOpenAIApi(FormulaUnderstandingAnalyserAsync):
    """Configuration for OpenAI-compatible API client."""

    url: str = 'http://localhost:8000/v1/chat/completions'
    headers: dict[str, str] = field(default_factory=lambda: {})
    params: dict[str, Any] = field(default_factory=lambda: {})
    timeout: float = 60 * 1  # 1 minute
    prompt: str = "Extract the LaTeX code from this image. Output only the LaTeX formula, nothing else. If no mathematical formula is found, output nothing. If more than one formula, then output all seperated by new lines. Don't miss any detail please."

    def encode_image_to_base64(self, image: Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    async def extract_formula_from_img(self, image: Image) -> str:
        """Send image to OpenAI-compatible API and extract formula."""
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
                        logger.info(f'Formula extraction successful: {content}...')
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


@dataclass(slots=True)
class FormulaUnderstandingAnalyserCli(FormulaUnderstandingAnalyserAsync):
    subprocess_kwargs: dict[str, Any] = field(default_factory=lambda: {})

    @abstractmethod
    def generate_command(self, img_path: str) -> list[str]:
        raise NotImplementedError

    def format_output(self, output: str) -> str:
        return output

    async def extract_formula_from_img(self, image: Image) -> str:
        try:
            with NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file.name, format='PNG')
                tmp_path = tmp_file.name

            try:
                cmd = self.generate_command(tmp_path)

                logger.info('Executing command: ' + ' '.join(cmd))

                result = subprocess.run(
                    cmd, capture_output=True, text=True, **self.subprocess_kwargs
                )

                if result.returncode == 0:
                    logger.debug(f'CLI Output: {result.stdout}')
                    formula_text = self.format_output(result.stdout).strip()
                    logger.info(f'Extraction successful: {formula_text}...')
                    return formula_text
                else:
                    logger.error(f'Error: {result.stderr}')
                    return ''

            finally:
                Path(tmp_path).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            logger.error('Error: timeout')
            return ''
        except Exception as e:
            logger.error(f'Error: {e}')
            return ''


@dataclass(slots=True)
class FormulaUnderstandingAnalyserTesseractCli(FormulaUnderstandingAnalyserCli):
    """Configuration for Tesseract CLI."""

    cmd: str = 'tesseract'
    lang: str = 'eng+equ'
    config_options: str = ''
    timeout: float = 30.0

    def __post_init__(self):
        self.subprocess_kwargs['timeout'] = self.timeout

    def generate_command(self, img_path: str) -> list[str]:
        cmd = [
            self.cmd,
            '-l',
            self.lang,
            img_path,
            'stdout',
        ]

        if self.config_options:
            cmd.extend(self.config_options.split())
        return cmd


@dataclass(slots=True)
class FormulaUnderstandingAnalyserPix2TexCli(FormulaUnderstandingAnalyserCli):
    """Configuration for Pix2Tex CLI."""

    cmd: str = 'pix2tex'
    timeout: float = 30.0

    def __post_init__(self):
        self.subprocess_kwargs['timeout'] = self.timeout

    def generate_command(self, img_path: str) -> list[str]:
        cmd = [self.cmd, img_path]
        return cmd

    def format_output(self, output: str) -> str:
        _, output = output.split(': ', 1)
        return output


@dataclass(slots=True)
class FormulaUnderstandingAnalyserRapidLatexOCRCli(FormulaUnderstandingAnalyserCli):
    """Configuration for RapidLatexOCR CLI."""

    cmd: str = 'rapid_latex_ocr'
    timeout: float = 30.0

    def __post_init__(self):
        self.subprocess_kwargs['timeout'] = self.timeout

    def generate_command(self, img_path: str) -> list[str]:
        cmd = [self.cmd, img_path]
        return cmd

    def format_output(self, output: str) -> str:
        lines = output.splitlines()
        if len(lines) < 2:
            return output
        lines.pop(-1)  # remove last line as it has `cost: <cost>`
        latex = '\n'.join(lines)
        try:
            return deduplicate_latex(latex)
        except Exception as e:
            logger.error(f'Got error when trying to deduplicate latex: {latex}: {e}')
            return '\n'.join(lines)


class FormulaUnderstandingAnalyserEnrichmentModel(BaseItemAndImageEnrichmentModel):
    images_scale = 2.6

    def __init__(self, formula_analyser: FormulaUnderstandingAnalyser):
        self.formula_analyser = formula_analyser

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return isinstance(element, TextItem) and element.label == DocItemLabel.FORMULA

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        yield from self.formula_analyser.analyse(doc, element_batch)


def formula_understanding_analyser_pipeline_cls(
    paginated_pipeline_cls: type[PaginatedPipeline],
    formula_analyser: FormulaUnderstandingAnalyser,
) -> type:
    class FormulaUnderstandingAnalyserPipeline(paginated_pipeline_cls):
        def __init__(self, pipeline_options: PaginatedPipelineOptions):
            super().__init__(pipeline_options)
            self.enrichment_pipe = (self.enrichment_pipe or []) + [
                FormulaUnderstandingAnalyserEnrichmentModel(formula_analyser=formula_analyser)
            ]

            self.keep_backend = True

    return FormulaUnderstandingAnalyserPipeline
