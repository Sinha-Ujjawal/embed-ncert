from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from docling.datamodel.pipeline_options import (EasyOcrOptions, OcrOptions,
                                                RapidOcrOptions,
                                                TesseractCliOcrOptions,
                                                TesseractOcrOptions)


class OcrEngine(StrEnum):
    EASY_OCR = "easy_ocr"
    RAPID_OCR = "rapid_ocr"
    TESSERACT_OCR = "tesseract_ocr"
    TESSERACT_CLI_OCR = "tesseract_cli_ocr"


@dataclass
class OCRConfig:
    languages: list[str]
    force_full_page_ocr: bool
    ocr_engine: OcrEngine
    ocr_engine_conf: dict[str, Any] = field(default_factory=lambda: {})

    def docling_ocr_options(self) -> OcrOptions:
        if self.ocr_engine == OcrEngine.EASY_OCR:
            cls = EasyOcrOptions
        elif self.ocr_engine == OcrEngine.RAPID_OCR:
            cls = RapidOcrOptions
        elif self.ocr_engine == OcrEngine.TESSERACT_OCR:
            cls = TesseractOcrOptions
        elif self.ocr_engine == OcrEngine.TESSERACT_CLI_OCR:
            cls = TesseractCliOcrOptions
        else:
            raise NotImplementedError(
                f"OCR Engine: `{self.ocr_engine}` not implemented yet!"
            )
        return cls(
            lang=self.languages,
            force_full_page_ocr=self.force_full_page_ocr,
            **self.ocr_engine_conf,
        )
