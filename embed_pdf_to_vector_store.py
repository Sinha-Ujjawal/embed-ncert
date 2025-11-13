import argparse
import hashlib
import uuid
from glob import glob

from load_dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv(override=True)


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

    from docling.chunking import HybridChunker
    from langchain_core.documents import Document
    from langchain_docling.loader import MetaExtractor

    from app_config import AppConfig

    app_config = AppConfig.from_yaml(args.conf)
    print(f'{app_config=}')
    converter = app_config.docling_config.docling_pdf_converter()
    file_paths = glob(args.pdf)
    embeddings = app_config.embedding_config.langchain_embedding()
    if app_config.tokenizer_config:
        chunker = HybridChunker(tokenizer=app_config.tokenizer_config.docling_tokenizer())
    else:
        chunker = HybridChunker()
    meta_extractor = MetaExtractor()
    vector_store = app_config.vector_store_config.get_vectorstore(embeddings)
    for file_path in tqdm(file_paths):
        print(f'Embedding file: {file_path}')
        document = converter.convert(file_path).document
        chunks = []
        for chunk_id, chunk in tqdm(enumerate(chunker.chunk(document), 1)):
            chunk_id = make_doc_id(f'{file_path}-chunk-{chunk_id}')
            chunks.append(
                Document(
                    page_content=chunker.contextualize(chunk=chunk),
                    metadata={
                        **meta_extractor.extract_chunk_meta(
                            file_path=file_path,
                            chunk=chunk,
                        ),
                        'chunk_id': chunk_id,
                    },
                    id=chunk_id,
                )
            )
        vector_store.add_documents(chunks)


if __name__ == '__main__':
    main()
