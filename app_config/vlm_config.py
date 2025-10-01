from dataclasses import dataclass
from functools import cached_property

from docling.datamodel.pipeline_options import PictureDescriptionVlmOptions


@dataclass
class VLMConfig:
    repo_id: str
    prompt: str

    @cached_property
    def docling_picture_description_options(self) -> PictureDescriptionVlmOptions:
        return PictureDescriptionVlmOptions(repo_id=self.repo_id, prompt=self.prompt)
