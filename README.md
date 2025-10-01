# Embed PDFs
This repo is for exploring how to convert PDF documents to enriched markdown using [docling](https://docling-project.github.io/docling/), and then how to integrate it with [langchain\_docling](https://python.langchain.com/docs/integrations/document_loaders/docling/) for creating [vector stores](https://en.wikipedia.org/wiki/Vector_database) for [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) pipeline. This is purely for learning and exploration.

## Getting Started with [Docker](https://www.docker.com/)

1. Build image using [Dockerfile](./Dockerfile). Provide `<input-name>`.

```console
docker build -t <image-name> .
```

2. Run below command to convert `pdf` to `markdown`. Provide `<image-name>`, `<lang>` and `<input-pdf>`. Below command assumes that your pdf is in `./data` folder. If you want to specify a different path, then make sure to mount the volume before calling the python file.

```console
docker run -v "$(pwd)/data/:/app/data/" --rm <image-name> .venv/bin/python pdf_to_md.py --conf conf/app/<lang>.yaml --pdf data/<input-pdf>
```

## Copyrights

Licensed under [@MIT](./LICENSE)
