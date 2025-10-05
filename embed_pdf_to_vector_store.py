import argparse
import hashlib
import uuid
from glob import glob

from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_docling.loader import MetaExtractor
from load_dotenv import load_dotenv

from app_config import AppConfig

load_dotenv()


def make_doc_id(content: str) -> str:
    # Hash content first (to normalize size)
    digest = hashlib.md5(content.encode()).hexdigest()
    # Convert hash into a UUID
    return str(uuid.UUID(digest[:32]))  # take first 32 hex chars


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Helper script to embed pdf document to vector store'
    )
    parser.add_argument(
        '--conf',
        required=True,
        help='Config yaml file to use. See conf/application folder for examples',
    )
    parser.add_argument(
        '--pdf',
        required=True,
        help='Path of the pdf file to process. Note that glob path is also supported.',
    )
    args = parser.parse_args()
    app_config = AppConfig.from_yaml(args.conf)
    print(f'{app_config=}')
    converter = app_config.docling_config.docling_pdf_converter()
    file_paths = glob(args.pdf)
    chunker = HybridChunker()
    meta_extractor = MetaExtractor()
    embeddings = app_config.embedding_config.langchain_embedding()
    for file_path in file_paths:
        print(f'Embedding file: {file_path}')
        document = converter.convert(file_path).document
        chunks = (
            Document(
                page_content=chunker.contextualize(chunk=chunk),
                metadata=meta_extractor.extract_chunk_meta(
                    file_path=file_path,
                    chunk=chunk,
                ),
                id=make_doc_id(f'{file_path}-chunk-{chunk_id}'),
            )
            for chunk_id, chunk in enumerate(chunker.chunk(document), 1)
        )
        _ = app_config.vector_store_config.from_documents(chunks, embeddings)


if __name__ == '__main__':
    main()
