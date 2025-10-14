# Embed PDFs
This repo is for exploring how to convert PDF documents to enriched markdown using [docling](https://docling-project.github.io/docling/), and then how to integrate it with [langchain\_docling](https://python.langchain.com/docs/integrations/document_loaders/docling/) for creating [vector stores](https://en.wikipedia.org/wiki/Vector_database) for [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) pipeline. This is purely for learning and exploration.

## Prerequisites

1. [uv](https://docs.astral.sh/uv/getting-started/)

2. [Qdrant Vector DB](https://qdrant.tech/)

## Getting Started locally

1. Download [uv](https://docs.astral.sh/uv/getting-started/) for managing python virtual environments. Note that this is not necessary, you can use python builtin venv as well.

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install all dependencies

```console
uv venv && .venv/bin/activate && uv sync --frozen
```

3. Install [tesseract-cli](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html). And then install tessdata for `hin`,  `eng` and `equ` from [here](https://github.com/tesseract-ocr/tessdata/tree/main).

4. Run [download\_models.sh](./download_models.sh) to install all models locally. By default, it will download to the `models` folder inside the root of the project. See [.env](.env) for the config.

5. Run [pdf\_to\_md.py](./pdf_to_md.py) to convert pdf to markdown.

6. Run [embed\_pdf\_to\_vector\_store.py](./embed_pdf_to_vector_store.py) to embed pdf document and store in qdrant vector db.

```console
.venv/bin/python pdf_to_md.py --conf conf/app/<lang>.yaml --pdf data/<input-pdf> --out <out>.pdf
```

## Getting Started with [Docker](https://www.docker.com/)

TODO

## Copyrights

Licensed under [@MIT](./LICENSE)
