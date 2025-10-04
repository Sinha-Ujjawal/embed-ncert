from dataclasses import dataclass
from pathlib import Path

from load_dotenv import load_dotenv
from omegaconf import OmegaConf
from utils.hydra_utils import instantiate_filtered

from app_config.docling_config import DoclingConfig
from app_config.embedding_config import EmbeddingConfig
from app_config.vector_store_config import VectorStoreConfig

DEFAULT_CONFIG = Path(__file__).parent.parent / 'conf/app/default.yaml'

load_dotenv()


@dataclass
class AppConfig:
    docling_config: DoclingConfig
    embedding_config: EmbeddingConfig
    vector_store_config: VectorStoreConfig

    @staticmethod
    def from_yaml(yaml_file: str) -> 'AppConfig':
        config = OmegaConf.merge(OmegaConf.load(DEFAULT_CONFIG), OmegaConf.load(yaml_file))
        obj = instantiate_filtered(config, _convert_='all')
        if not isinstance(obj, AppConfig):
            raise ValueError(
                f'Cannot instantiate the AppConfig object using `{DEFAULT_CONFIG}` + `{yaml_file}`'
            )
        return obj
