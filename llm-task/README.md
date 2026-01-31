# PNU CHALLENGE : LLM TASK

## CVE - CWE and Personas Data Preparation
```bash
uv run pipelines/download_datasets.py
uv run pipelines/prepare_data.py
```

## CVE - CWE and Personas Data Vector Embedding & Ingestion
```bash
uv run pipelines/ingest.py
```

## Run RAG and LLM Server
```bash
uv run uvicorn api:app --reload
```

## RAG and LLM Testing
```bash
uv run test_rag.py
uv run test_benchmark.py
```


