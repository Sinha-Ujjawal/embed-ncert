from dataclasses import dataclass
from pathlib import Path

from load_dotenv import load_dotenv
from mem0.configs.base import MemoryConfig
from omegaconf import OmegaConf
from utils.hydra_utils import instantiate_filtered

from app_config.docling_config import DoclingConfig
from app_config.embedding_config import EmbeddingConfig
from app_config.tokenizer_config import TokenizerConfig
from app_config.vector_store_config import VectorStoreConfig

PROJECT_ROOT_PATH = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT_PATH / 'conf/app/default.yaml'

load_dotenv()


def read_file_resolver(path: str) -> str:
    return (PROJECT_ROOT_PATH / path).read_text()


OmegaConf.register_new_resolver('read_file', read_file_resolver)


@dataclass
class AppConfig:
    docling_config: DoclingConfig
    embedding_config: EmbeddingConfig
    vector_store_config: VectorStoreConfig
    reranking_embedding_config: EmbeddingConfig
    mem0_config: MemoryConfig
    tokenizer_config: TokenizerConfig | None = None

    @staticmethod
    def from_yaml(yaml_file: str) -> 'AppConfig':
        config = OmegaConf.merge(OmegaConf.load(DEFAULT_CONFIG), OmegaConf.load(yaml_file))
        obj = instantiate_filtered(config, _convert_='all')
        if not isinstance(obj, AppConfig):
            raise ValueError(
                f'Cannot instantiate the AppConfig object using `{DEFAULT_CONFIG}` + `{yaml_file}`'
            )
        return obj
