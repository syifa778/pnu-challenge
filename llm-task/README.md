# Retrieval Augmented Generation (RAG) System for Cybersecurity CVE and Personal Information
This work presents the design, implementation, and evaluation of Retrieval Augment Generation (RAG) system integrated with Large Language Model (LLM). The system utilizes two distinct datasets: cybersecurity vulnerability data (CVE/CWE) and personal information. While the RAG architecture is designed to retrieve data from both sources, a primary constraint is ensuring the LLM does not leak sensitive personal information within its generated outputs.

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


