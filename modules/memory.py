import json
import os

memory_file = "./memory/memory.jsonl"

# memory content format 
# { "event": "message|action|event_type", "content": "message|action|event", "timestamp": "timestamp", "user": "user|system|assistant", "username": "username|system|event_name"}

def append_jsonl(data, path=memory_file):
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


