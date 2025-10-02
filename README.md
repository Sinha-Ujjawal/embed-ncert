# Embed PDFs
This repo is for exploring how to convert PDF documents to enriched markdown using [docling](https://docling-project.github.io/docling/), and then how to integrate it with [langchain\_docling](https://python.langchain.com/docs/integrations/document_loaders/docling/) for creating [vector stores](https://en.wikipedia.org/wiki/Vector_database) for [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) pipeline. This is purely for learning and exploration.

## Getting Started locally

1. Download [uv](https://docs.astral.sh/uv/getting-started/) for managing python virtual environments. Note that this is not necessary, you can use python builtin venv as well.

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install all dependencies

```console
uv venv && .venv/bin/activate && uv sync --frozen
```

3. Install [tesseract-cli](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html). And then install tessdata for `hin` and `eng` from [here](https://github.com/tesseract-ocr/tessdata/tree/main).

4. Run [download\_models.sh](./download_models.sh) to install all models locally. By default, it will download to the `models` folder inside the root of the project. See [.env](.env) for the config.

5. Run [pdf\_to\_md.py](./pdf_to_md.py) to convert pdf to markdown.

```console
.venv/bin/python pdf_to_md.py --conf conf/app/<lang>.yaml --pdf data/<input-pdf> --out <out>.pdf
```

## Getting Started with [Docker](https://www.docker.com/)

1. Build image using [Dockerfile](./Dockerfile). Provide `<input-name>`.

```console
docker build -t <image-name> .
```

2. Run below command to convert `pdf` to `markdown`. Provide `<image-name>`, `<lang>` and `<input-pdf>`. Below command assumes that your pdf is in `./data` folder. If you want to specify a different path, then make sure to mount the volume before calling the python file.

```console
docker run -v "$(pwd)/data/:/app/data/" --rm <image-name> .venv/bin/python pdf_to_md.py --conf conf/app/<lang>.yaml --pdf data/<input-pdf> --out <out>.pdf
```

## Help for [pdf\_to\_md.py](./pdf_to_md.py)

```console
> .venv/bin/python pdf_to_md.py --help
usage: pdf_to_md.py [-h] --conf CONF --pdf PDF [--out OUT]

Helper script to convert pdf document to Docling document

options:
  -h, --help   show this help message and exit
  --conf CONF  Config yaml file to use. See conf/application folder for examples
  --pdf PDF    Path of the pdf file to process
  --out OUT    Path of the output markdown file
```

## Copyrights

Licensed under [@MIT](./LICENSE)
