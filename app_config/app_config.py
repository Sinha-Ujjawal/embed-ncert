from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from hydra.utils import instantiate
from omegaconf import OmegaConf

from app_config.ocr_config import OCRConfig
from app_config.vlm_config import VLMConfig

DEFAULT_CONFIG = Path(__file__).parent.parent / "conf/app/default.yaml"


@dataclass
class AppConfig:
    ocr_config: OCRConfig | None
    vlm_config: VLMConfig | None
    generate_page_images: bool
    images_scale: float
    # Enrichments
    do_formula_enrichment: bool  # perform formula OCR, return Latex code
    do_table_structure: bool
    do_code_enrichment: bool

    @staticmethod
    def from_yaml(yaml_file: str) -> "AppConfig":
        config = OmegaConf.merge(
            OmegaConf.load(DEFAULT_CONFIG), OmegaConf.load(yaml_file)
        )
        obj = instantiate(config, _convert_="all")
        if not isinstance(obj, AppConfig):
            raise ValueError(
                f"Cannot instantiate the AppConfig object using `{DEFAULT_CONFIG}` + `{yaml_file}`"
            )
        return obj

    @cached_property
    def docling_pdf_pipeline_options(self) -> PdfPipelineOptions:
        options = PdfPipelineOptions(do_ocr=False)
        if self.ocr_config:
            options.do_ocr = True
            options.ocr_options = self.ocr_config.docling_ocr_options
        if self.vlm_config:
            options.do_picture_description = True
            options.do_picture_classification = True
            options.picture_description_options = (
                self.vlm_config.docling_picture_description_options
            )
        options.generate_page_images = self.generate_page_images
        options.images_scale = self.images_scale
        # Enrichments
        options.do_formula_enrichment = self.do_formula_enrichment
        options.do_table_structure = self.do_table_structure
        options.do_code_enrichment = self.do_code_enrichment
        return options

    @cached_property
    def docling_pdf_converter(self) -> DocumentConverter:
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.docling_pdf_pipeline_options
                )
            }
        )
