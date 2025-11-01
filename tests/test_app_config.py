from glob import glob
from pathlib import Path

from app_config import AppConfig

PROJECT_ROOT = Path(__file__).parent.parent


def test_create_app_config_from_yaml_files():
    for yaml_file in glob(f'{PROJECT_ROOT}/conf/app/*.yaml'):
        AppConfig.from_yaml(yaml_file)
