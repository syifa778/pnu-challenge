from collections import defaultdict
from core.safety import redact_pii

# session_id -> list[str]
MEMORY_STORE = defaultdict(list)

MAX_MEMORY_ITEMS = 10

def get_memory(session_id: str):
    return MEMORY_STORE[session_id]

def append_memory(session_id: str, text: str):
    clean = redact_pii(text)
    MEMORY_STORE[session_id].append(clean)

    # trim
    MEMORY_STORE[session_id] = MEMORY_STORE[session_id][-MAX_MEMORY_ITEMS:]
