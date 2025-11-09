## TODOS
- Implement pagination in `/history`
- Implement user auth
- Only allow single request per thread\_id
- ...

## DONE
See [embed\_pdf\_to\_vector\_store.py](./embed_pdf_to_vector_store.py)
- Raw Document -> Docling Document
- Docling Document -> Qdrant Vector DB

See [agent.py](./agent.py)
- Simple RAG Agent
- Implement tracing via [mlflow](https://mlflow.org/docs/3.0.1/tracing/)
- Implement memory using [mem0](https://github.com/mem0ai/mem0)

See [server.py](./server.py)
- Implement POST `/query` for Question Answering
- Implement GET '/history` for getting historical messages
