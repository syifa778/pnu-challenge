uv run pipelines/download_datasets.py
uv run pipelines/prepare_data.py
uv run pipelines/ingest.py
uv run uvicorn api:app --reload