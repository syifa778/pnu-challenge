import json
from pathlib import Path
from typing import List, Dict


DATASET_DIR = Path("data")


def load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_documents_jsonl(documents: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def prepare_cve_documents(cve_data: List[Dict]) -> List[Dict]:
    documents = []

    for item in cve_data:
        text = f"""
            CVE ID: {item.get('CVE-ID')}, Severity: {item.get('SEVERITY')}, CVSS v4: {item.get('CVSS-V4')}, CVSS v3: {item.get('CVSS-V3')}, CVSS v2: {item.get('CVSS-V2')}, CWE ID: {item.get('CWE-ID')}, Description:, {item.get('DESCRIPTION')}""".strip()

        metadata = {
            "type": "cve",
            "source": "cve",
            "cve_id": item.get("CVE-ID"),
            "severity": item.get("SEVERITY"),
            "cwe_id": item.get("CWE-ID"),
        }

        documents.append({
            "text": text,
            "metadata": metadata
        })

    return documents


def prepare_persona_documents(persona_data: List[Dict]) -> List[Dict]:
    documents = []

    for item in persona_data:
        text = f"""
            General Persona: {item.get('persona')}, Professional Persona: {item.get('professional_persona')}, Career Goals: {item.get('career_goals_and_ambitions')}, Skills and Expertise: {item.get('skills_and_expertise')}, Hobbies and Interests: {item.get('hobbies_and_interests')}, Location: {item.get('city')}, {item.get('state')}, {item.get('country')}""".strip()

        metadata = {
            "type": "persona",
            "source": "persona",
            "uuid": item.get("uuid"),
            "age": item.get("age"),
            "sex": item.get("sex"),
            "city": item.get("city"),
            "state": item.get("state"),
            "country": item.get("country"),
            "occupation": item.get("occupation"),
        }

        documents.append({
            "text": text,
            "metadata": metadata
        })

    return documents


def prepare_all_documents() -> List[Dict]:
    cve_path = DATASET_DIR / "raw/cve.jsonl"
    persona_path = DATASET_DIR / "raw/personas.jsonl"

    documents = []

    if cve_path.exists():
        cve_data = load_jsonl(cve_path)
        documents.extend(prepare_cve_documents(cve_data))

    if persona_path.exists():
        persona_data = load_jsonl(persona_path)
        documents.extend(prepare_persona_documents(persona_data))

    return documents


if __name__ == "__main__":
    docs = prepare_all_documents()
    output_path = DATASET_DIR / "processed/documents.jsonl"
    save_documents_jsonl(docs, output_path)
    print(f"Prepared {len(docs)} documents")
