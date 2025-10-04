import argparse
from pathlib import Path

from load_dotenv import load_dotenv

from app_config import AppConfig

load_dotenv()


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Helper script to convert pdf document to Docling document'
    )
    parser.add_argument(
        '--conf',
        required=True,
        help='Config yaml file to use. See conf/application folder for examples',
    )
    parser.add_argument('--pdf', required=True, help='Path of the pdf file to process')
    parser.add_argument('--out', required=False, help='Path of the output markdown file')
    args = parser.parse_args()
    app_config = AppConfig.from_yaml(args.conf)
    print(f'{app_config=}')
    opts = app_config.docling_config.docling_paginated_pipeline_cls_and_options()
    print(f'{opts=}')
    converter = app_config.docling_config.docling_pdf_converter()
    document = converter.convert(args.pdf).document
    if args.out:
        print(f'Writing to {args.out}')
        Path(args.out).write_text(document.export_to_markdown())
    else:
        print(document.export_to_markdown())
    print('Done')


if __name__ == '__main__':
    main()
