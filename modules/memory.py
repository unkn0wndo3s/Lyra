import json
import os
from typing import TypedDict

memory_file = "./memory/memory.jsonl"

# memory content format 
# { "event": "message|action|event_type", "content": "message|action|event", "timestamp": "timestamp", "user": "user|system|assistant", "username": "username|system|event_name"}

class MemoryContent(TypedDict):
    event: str  # "message|action|event_type"
    content: str  # "message|action|event"
    timestamp: str  # "timestamp"
    user: str  # "user|system|assistant"
    username: str  # "username|system|event_name"

def append_jsonl(data: MemoryContent, path: str = memory_file) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(data, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def read_jsonl(path=memory_file):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


