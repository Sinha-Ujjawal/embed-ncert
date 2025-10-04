from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PaginatedPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from load_dotenv import load_dotenv
from omegaconf import OmegaConf
from utils.hydra_utils import instantiate_filtered

from app_config.formula_understanding_pipeline import (
    FormulaUnderstandingAnalyser,
    formula_understanding_analyser_pipeline_cls,
)
from app_config.ocr_config import OCRConfig
from app_config.picture_desc_config import (
    PictureDescriptionApiOptions,
    PictureDescriptionConfig,
)

DEFAULT_CONFIG = Path(__file__).parent.parent / 'conf/app/default.yaml'

load_dotenv()


@dataclass
class AppConfig:
    ocr_config: OCRConfig
    generate_page_images: bool = True
    images_scale: float = 1.0
    do_picture_classification: bool = True
    # Enrichments
    picture_desc_config: PictureDescriptionConfig | None = None
    formula_analyser: FormulaUnderstandingAnalyser | None = None

    @staticmethod
    def from_yaml(yaml_file: str) -> 'AppConfig':
        config = OmegaConf.merge(OmegaConf.load(DEFAULT_CONFIG), OmegaConf.load(yaml_file))
        obj = instantiate_filtered(config, _convert_='all')
        if not isinstance(obj, AppConfig):
            raise ValueError(
                f'Cannot instantiate the AppConfig object using `{DEFAULT_CONFIG}` + `{yaml_file}`'
            )
        return obj

    def docling_paginated_pipeline_cls_and_options(self) -> tuple[type, PaginatedPipelineOptions]:
        cls, options = self.ocr_config.docling_paginated_pipeline_cls_and_options()
        options.generate_page_images = self.generate_page_images
        options.images_scale = self.images_scale
        options.do_picture_classification = self.do_picture_classification
        if self.picture_desc_config:
            options.do_picture_description = True
            options.do_picture_classification = True
            options.picture_description_options = (
                self.picture_desc_config.docling_picture_description_options()
            )
            if isinstance(options.picture_description_options, PictureDescriptionApiOptions):
                options.enable_remote_services = True
        if self.formula_analyser:
            cls = formula_understanding_analyser_pipeline_cls(cls, self.formula_analyser)
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
