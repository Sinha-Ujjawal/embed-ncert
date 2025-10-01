import argparse

from load_dotenv import load_dotenv

from app_config import AppConfig

load_dotenv()


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Helper script to convert pdf document to Docling document"
    )
    parser.add_argument(
        "--conf",
        required=True,
        help="Config yaml file to use. See conf/application folder for examples",
    )
    parser.add_argument("--pdf", required=True, help="Path of the pdf file to process")
    args = parser.parse_args()
    app_config = AppConfig.from_yaml(args.conf)
    opts = app_config.docling_pdf_pipeline_options
    print(f"{opts=}")
    converter = app_config.docling_pdf_converter()
    document = converter.convert(args.pdf).document
    print(document.export_to_markdown())


if __name__ == "__main__":
    main()
