from dataclasses import dataclass

from app_config.docling_config.formula_understanding_pipeline import (
    FormulaUnderstandingAnalyser,
    formula_understanding_analyser_pipeline_cls,
)
from app_config.docling_config.ocr_config import OCRConfig
from app_config.docling_config.picture_desc_pipeline import (
    PictureDescAnalyser,
    picture_desc_analyser_pipeline_cls,
)
from app_config.docling_config.text_enhancer_pipeline import (
    TextEnhancerAnalyser,
    text_enhancer_analyser_pipeline_cls,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PaginatedPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


@dataclass
class DoclingConfig:
    ocr_config: OCRConfig
    generate_page_images: bool = True
    generate_picture_images: bool = False
    images_scale: float = 1.0
    do_picture_classification: bool = True
    # Enrichments
    picture_desc_analyser: PictureDescAnalyser | None = None
    formula_analyser: FormulaUnderstandingAnalyser | None = None
    text_analyser: TextEnhancerAnalyser | None = None

    def docling_paginated_pipeline_cls_and_options(self) -> tuple[type, PaginatedPipelineOptions]:
        cls, options = self.ocr_config.docling_paginated_pipeline_cls_and_options()
        options.generate_page_images = self.generate_page_images
        options.generate_picture_images = self.generate_picture_images
        options.images_scale = self.images_scale
        options.do_picture_classification = self.do_picture_classification
        if self.picture_desc_analyser:
            cls = picture_desc_analyser_pipeline_cls(cls, self.picture_desc_analyser)
        if self.formula_analyser:
            cls = formula_understanding_analyser_pipeline_cls(cls, self.formula_analyser)
        if self.text_analyser:
            cls = text_enhancer_analyser_pipeline_cls(cls, self.text_analyser)
        return cls, options

    def docling_pdf_converter(self) -> DocumentConverter:
        pipeline_cls, pipeline_options = self.docling_paginated_pipeline_cls_and_options()
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=pipeline_cls,
                    pipeline_options=pipeline_options,
                )
            }
        )
