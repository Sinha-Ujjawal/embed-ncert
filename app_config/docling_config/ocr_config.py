import os
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrOptions,
    PaginatedPipelineOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    AnyUrl,
    ApiVlmOptions,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline
from load_dotenv import load_dotenv

load_dotenv()

DOCLING_ARTIFACTS_PATH = os.environ.get('DOCLING_ARTIFACTS_PATH')


@dataclass(slots=True)
class OCRConfig:
    @abstractmethod
    def docling_paginated_pipeline_cls_and_options(self) -> tuple[type, PaginatedPipelineOptions]:
        raise NotImplementedError


@dataclass(slots=True)
class InlineVlmConfig:
    repo_id: str
    response_format: ResponseFormat
    addnl_conf: dict[str, Any] = field(default_factory=lambda: {})

    def vlm_options(self, *args, **kwargs) -> InlineVlmOptions:
        return InlineVlmOptions(
            *args,
            **kwargs,
            repo_id=self.repo_id,
            response_format=self.response_format,
            **self.addnl_conf,
        )


@dataclass(slots=True)
class ApiVlmConfig:
    url: AnyUrl
    params: dict[str, Any]
    response_format: ResponseFormat
    concurrency: int = 1
    timeout: int = 180
    addnl_conf: dict[str, Any] = field(default_factory=lambda: {})

    def vlm_options(self, *args, **kwargs) -> ApiVlmOptions:
        return ApiVlmOptions(
            *args,
            **kwargs,
            url=self.url,
            params=self.params,
            response_format=self.response_format,
            concurrency=self.concurrency,
            timeout=self.timeout,
            **self.addnl_conf,
        )


@dataclass(slots=True)
class VlmOCRConfig(OCRConfig):
    prompt: str
    vlm_config: InlineVlmConfig | ApiVlmConfig
    scale: float = 2.0
    max_size: int | None = None
    temperature: float = 0.0

    def docling_paginated_pipeline_cls_and_options(self) -> tuple[type, VlmPipelineOptions]:
        options = VlmPipelineOptions()
        options.artifacts_path = DOCLING_ARTIFACTS_PATH
        options.vlm_options = self.vlm_config.vlm_options(
            prompt=self.prompt,
            scale=self.scale,
            max_size=self.max_size,
            temperature=self.temperature,
        )
        return VlmPipeline, options


class StandardOcrEngine(StrEnum):
    EASY_OCR = 'easy_ocr'
    RAPID_OCR = 'rapid_ocr'
    TESSERACT_OCR = 'tesseract_ocr'
    TESSERACT_CLI_OCR = 'tesseract_cli_ocr'


@dataclass(slots=True)
class StandardOCRConfig(OCRConfig):
    languages: list[str]
    force_full_page_ocr: bool
    ocr_engine: StandardOcrEngine
    ocr_engine_conf: dict[str, Any] = field(default_factory=lambda: {})
    # Enrichments
    do_formula_enrichment: bool = False  # perform formula OCR, return Latex code
    do_table_structure: bool = False
    do_code_enrichment: bool = False

    def docling_ocr_options(self) -> OcrOptions:
        if self.ocr_engine == StandardOcrEngine.EASY_OCR:
            cls = EasyOcrOptions
        elif self.ocr_engine == StandardOcrEngine.RAPID_OCR:
            cls = RapidOcrOptions
        elif self.ocr_engine == StandardOcrEngine.TESSERACT_OCR:
            cls = TesseractOcrOptions
        elif self.ocr_engine == StandardOcrEngine.TESSERACT_CLI_OCR:
            cls = TesseractCliOcrOptions
        else:
            raise NotImplementedError(f'OCR Engine: `{self.ocr_engine}` not implemented yet!')
        return cls(
            lang=self.languages,
            force_full_page_ocr=self.force_full_page_ocr,
            **self.ocr_engine_conf,
        )

    def docling_paginated_pipeline_cls_and_options(self) -> tuple[type, PdfPipelineOptions]:
        options = PdfPipelineOptions()
        options.artifacts_path = DOCLING_ARTIFACTS_PATH
        options.do_ocr = True
        options.ocr_options = self.docling_ocr_options()
        # Enrichments
        options.do_formula_enrichment = self.do_formula_enrichment
        options.do_table_structure = self.do_table_structure
        options.do_code_enrichment = self.do_code_enrichment
        return StandardPdfPipeline, options
