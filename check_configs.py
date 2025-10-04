from glob import glob

from app_config import AppConfig

for yaml_file in glob('./conf/app/*.yaml'):
    print(f'Trying to make app config for {yaml_file=}')
    app_config = AppConfig.from_yaml(yaml_file)
