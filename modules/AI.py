from multiprocessing import context
from typing import List, Dict, Optional
from memory import MemoryContent
import time
try:
    import ollama
    from ollama import ResponseError
except Exception as e:
    raise ImportError("the package 'ollama' is required. Install it with: 'pip install ollama'") from e
model="P2Wdisabled/lyra:7b"
def send_history(messages: List[Dict], context: str, model: str = model, host: Optional[str] = None, retry_pull: bool = True) -> str:
    if not isinstance(messages, list):
        raise TypeError("messages must be a list of objects")
    if not isinstance(context, str):
        raise TypeError("context must be a string")
    
    client_kwargs = {}
    if host:
        client_kwargs["host"] = host
    
    client = ollama.Client(**client_kwargs)
    if hasattr(ollama, "Client"):
        system_msg = {"role": "system", "content": context}
        combined = [system_msg] + messages

        def _call_chat():
            if client:
                return client.chat(model=model, messages=combined)
            else:
                return ollama.chat(model=model, messages=combined)
    
        try:
            response = _call_chat()
            try:
                return response["message"]["content"]
            except Exception:
                try:
                    return response.message.content
                except Exception:
                    return str(response)
        except ResponseError as e:
            status = getattr(e, "status_code", None)
            if retry_pull and (status == 404 or "not found" in str(e).lower()):
                try:
                    ollama.pull(model)
                    time.sleep(1.0)
                    response = _call_chat()
                    try:
                        return response["message"]["content"]
                    except Exception:
                        try:
                            return response.message.content
                        except Exception:
                            return str(response)
                except Exception as e2:
                    raise RuntimeError(f"Failed after trying to pull model {model}: {e2}") from e2
            else:
                raise RuntimeError(f"Ollama error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error while calling the model: {e}") from e

#get the last 10 messages (from timestamp)
def get_history(memory: List[MemoryContent]) -> List[Dict]:
    history = []
    for item in memory:
        if item["event"] == "message":
            history.append({"role": item["role"], "content": item["content"]})
    return history[-10:] if len(history) > 10 else history