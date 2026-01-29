from datasets import load_dataset
import json
import os

os.makedirs("data/raw", exist_ok=True)

# =====================
# 1. CVE DATA
# =====================
cve_ds = load_dataset(
    "stasvinokur/cve-and-cwe-dataset-1999-2025",
    split="train[-200:]",
    cache_dir="./datasets"
)

with open("data/raw/cve.jsonl", "w", encoding="utf-8") as f:
    for row in cve_ds:
        f.write(json.dumps(dict(row)) + "\n")

print("CVE dataset (200 samples, all columns) saved")

# =========================================
# 2. PERSONA DATA (ALL COLUMNS)
# =========================================
persona_ds = load_dataset(
    "nvidia/Nemotron-Personas-USA",
    split="train[:100]",
    cache_dir="./datasets"
)

with open("data/raw/personas.jsonl", "w", encoding="utf-8") as f:
    for row in persona_ds:
        f.write(json.dumps(dict(row)) + "\n")

print("Persona dataset (100 samples, all columns) saved")
