import json
import os
import time
from typing import TypedDict, List
import AI as AI

memory_file = "./memory/memory.jsonl"

# memory content format 
# { "event": "message|action|event_type", "content": "message|action|event", "timestamp": "timestamp", "user": "user|system|assistant", "username": "username|system|event_name"}

class MemoryContent(TypedDict):
    event: str  # "message|action|event_type"
    content: str  # "message|action|event"
    timestamp: str  # "timestamp"
    role: str  # "user|system|assistant"
    username: str  # "username|system|event_name"

def append_jsonl(data: MemoryContent, path: str = memory_file) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(data, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def read_jsonl(path=memory_file) -> List[MemoryContent]:
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

def messages_synthesis():
    all_messages = read_jsonl()
    messages = [msg for msg in all_messages if msg["event"] == "message"]
    synthesis_events = [msg for msg in all_messages if msg["event"] == "synthesis"]
    covered_timestamps = set()
    for synth in synthesis_events:
        if synth["username"].startswith("synthesis_"):
            try:
                parts = synth["username"].split("_")
                if len(parts) >= 3:
                    first_ts = float(parts[1])
                    last_ts = float(parts[2])
                    covered_timestamps.add((first_ts, last_ts))
            except (ValueError, IndexError):
                pass
    def is_covered(msg_ts: float) -> bool:
        for first_ts, last_ts in covered_timestamps:
            if first_ts <= msg_ts <= last_ts:
                return True
        return False
    unprocessed_messages = [msg for msg in messages if not is_covered(float(msg["timestamp"]))]
    for i in range(0, len(unprocessed_messages), 10):
        package = unprocessed_messages[i:i+10]
        if len(package) < 10:
            break
        ai_messages = [{"role": msg["role"], "content": msg["content"]} for msg in package]
        try:
            synthesis_content = AI.send_history(ai_messages, context="Summarize this conversation concisely, preserving important information and context.")
        except Exception as e:
            print(f"Error synthesizing messages: {e}")
            continue
        first_timestamp = float(package[0]["timestamp"])
        last_timestamp = float(package[-1]["timestamp"])
        synthesis_event: MemoryContent = {
            "event": "synthesis",
            "content": synthesis_content,
            "timestamp": str(time.time()),
            "role": "system",
            "username": f"synthesis_{first_timestamp}_{last_timestamp}"
        }
        append_jsonl(synthesis_event)
    all_synthesis = [msg for msg in read_jsonl() if msg["event"] == "synthesis"]
    if len(all_synthesis) >= 10:
        compressed_events = [msg for msg in read_jsonl() if msg["event"] == "compressed_memory"]
        covered_synth_timestamps = set()
        for comp in compressed_events:
            if comp["username"].startswith("compressed_"):
                try:
                    parts = comp["username"].split("_")
                    if len(parts) >= 3:
                        first_ts = float(parts[1])
                        last_ts = float(parts[2])
                        covered_synth_timestamps.add((first_ts, last_ts))
                except (ValueError, IndexError):
                    pass
        def is_synth_covered(synth_ts: float) -> bool:
            for first_ts, last_ts in covered_synth_timestamps:
                if first_ts <= synth_ts <= last_ts:
                    return True
            return False        
        unprocessed_synthesis = [synth for synth in all_synthesis if not is_synth_covered(float(synth["timestamp"]))]
        for i in range(0, len(unprocessed_synthesis), 10):
            package = unprocessed_synthesis[i:i+10]
            if len(package) < 10:
                break
            ai_messages = [{"role": "system", "content": synth["content"]} for synth in package]
            try:
                compressed_content = AI.send_history(ai_messages, context="Compress and summarize these synthesis summaries into a single comprehensive summary, preserving all important information descibing the conversation between the user and the assistant.", model="llama3.2:1b-text-fp16")
            except Exception as e:
                print(f"Error compressing synthesis: {e}")
                continue
            first_timestamp = float(package[0]["timestamp"])
            last_timestamp = float(package[-1]["timestamp"])
            compressed_event: MemoryContent = {
                "event": "compressed_memory",
                "content": compressed_content,
                "timestamp": str(time.time()),
                "role": "system",
                "username": f"compressed_{first_timestamp}_{last_timestamp}"
            }
            append_jsonl(compressed_event)

messages_synthesis()