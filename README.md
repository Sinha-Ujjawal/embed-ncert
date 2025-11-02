# Embed PDFs
This repo is for exploring how to convert PDF documents to enriched markdown using [docling](https://docling-project.github.io/docling/), and then how to integrate it with [langchain\_docling](https://python.langchain.com/docs/integrations/document_loaders/docling/) for creating [vector stores](https://en.wikipedia.org/wiki/Vector_database) for [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) pipeline. This is purely for learning and exploration.

## Prerequisites

1. [uv](https://docs.astral.sh/uv/getting-started/)

2. [Qdrant Vector DB](https://qdrant.tech/)

3. [Ollama](https://ollama.com/download)

4. [tesseract-cli](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)

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

4. Start Ollama server. If you don't have install it from [here](https://ollama.com/download), and then start ollama. On terminal you can

```console
OLLAMA_DEBUG=1 OLLAMA_CONTEXT_LENGTH=10000 ollama serve
```

5. Run [download\_models.sh](./download_models.sh) to install all models locally.
   - For Docling, it will download the docling models to the `models` folder inside the root of the project. See [.env](.env) for the config.
   - For Huggingface, it will download the models to `~/.cache/huggingface/hub/` folder
   - For Ollama, it will download the models to the OLLAMA\_MODELS directory. By default, it is `~/.ollama/models/`

6. Start Qdrant DB using Docker [Refer](https://qdrant.tech/documentation/quickstart/#download-and-run)

7. Run [embed\_pdf\_to\_vector\_store.py](./embed_pdf_to_vector_store.py) to embed pdf document and store in qdrant vector db.

```console
.venv/bin/python embed_pdf_to_vector_store.py --conf conf/app/<lang>.yaml --pdf data/<input-pdf>
```

8. Starting the [fastapi](https://fastapi.tiangolo.com/) server:
    - Run [mlflow](https://mlflow.org/) server for tracing
    ```console
    mlflow server
    ```
    - Run [server.py](./server.py) file using fastapi
    ```console
    fastapi run server.py
    ```
    - Then you can use visit [http://localhost:8000/docs](http://localhost:8000/docs) to see available endpoints
    - Refer [Probe.ipynb](./Probe.ipynb) to see example to use the api. Currently we have a way to ask question, and also continue on the thread

## Getting Started with [Docker](https://www.docker.com/)

TODO

## Copyrights

Licensed under [@MIT](./LICENSE)
