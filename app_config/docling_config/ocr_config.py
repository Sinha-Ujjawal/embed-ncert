from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from docling.datamodel.pipeline_options import AcceleratorDevice, EasyOcrOptions
from docling.datamodel.pipeline_options import LayoutModelConfig as DoclingLayoutModelConfig
from docling.datamodel.pipeline_options import (
    LayoutOptions,
    OcrOptions,
    PaginatedPipelineOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
    ThreadedPdfPipelineOptions,
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    AnyUrl,
    ApiVlmOptions,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline
from load_dotenv import load_dotenv

load_dotenv()


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
    generate_page_images: bool = True
    force_backend_text: bool = False

    def docling_paginated_pipeline_cls_and_options(self) -> tuple[type, VlmPipelineOptions]:
        options = VlmPipelineOptions()
        options.generate_page_images = self.generate_page_images
        options.force_backend_text = self.force_backend_text
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
class LayoutModelConfig:
    # model_spec params:
    name: str = 'docling_layout_heron'
    repo_id: str = 'ds4sd/docling-layout-heron'
    revision: str = 'main'
    model_path: str = ''
    supported_devices: list[AcceleratorDevice] = field(
        default_factory=lambda: [
            AcceleratorDevice.CPU,
            AcceleratorDevice.CUDA,
            AcceleratorDevice.MPS,
        ]
    )
    # layout_options params:
    create_orphan_clusters: bool = True  # Whether to create clusters for orphaned cells
    keep_empty_clusters: bool = False  # Whether to keep clusters that contain no text cells
    skip_cell_assignment: bool = False  # Skip cell-to-cluster assignment for VLM-only processing

    def docling_layout_options(self) -> LayoutOptions:
        model_spec = DoclingLayoutModelConfig(
            name=self.name,
            repo_id=self.repo_id,
            revision=self.revision,
            model_path=self.model_path,
            supported_devices=self.supported_devices,
        )
        return LayoutOptions(
            create_orphan_clusters=self.create_orphan_clusters,
            keep_empty_clusters=self.keep_empty_clusters,
            model_spec=model_spec,
            skip_cell_assignment=self.skip_cell_assignment,
        )


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
    layout_model_config: LayoutModelConfig = field(default_factory=lambda: LayoutModelConfig())

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

    def populate_options(self, options: PdfPipelineOptions) -> None:
        options.layout_options = self.layout_model_config.docling_layout_options()
        options.do_ocr = True
        options.ocr_options = self.docling_ocr_options()
        # Enrichments
        options.do_formula_enrichment = self.do_formula_enrichment
        options.do_table_structure = self.do_table_structure
        options.do_code_enrichment = self.do_code_enrichment

    def docling_paginated_pipeline_cls_and_options(self) -> tuple[type, PdfPipelineOptions]:
        options = PdfPipelineOptions()
        self.populate_options(options)
        return StandardPdfPipeline, options


@dataclass(slots=True)
class ThreadedStandardOCRConfig(StandardOCRConfig):
    # Batch sizes for different stages
    ocr_batch_size: int = 4
    layout_batch_size: int = 4
    table_batch_size: int = 4

    # Timing control
    batch_timeout_seconds: float = 2.0

    # Backpressure and queue control
    queue_max_size: int = 100

    def populate_options(self, options: PdfPipelineOptions) -> None:
        assert isinstance(options, ThreadedPdfPipelineOptions)
        StandardOCRConfig.populate_options(self, options)
        options.ocr_batch_size = self.ocr_batch_size
        options.layout_batch_size = self.layout_batch_size
        options.table_batch_size = self.table_batch_size
        options.batch_timeout_seconds = self.batch_timeout_seconds
        options.queue_max_size = self.queue_max_size

    def docling_paginated_pipeline_cls_and_options(self) -> tuple[type, ThreadedPdfPipelineOptions]:
        options = ThreadedPdfPipelineOptions()
        self.populate_options(options)
        return ThreadedStandardPdfPipeline, options
