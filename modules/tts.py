import tempfile
import os
from typing import Optional
def synthesize_text(text: str) -> bytes:
    try:
        import pyttsx3
    except Exception:
        raise RuntimeError("pyttsx3 not installed")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    engine = pyttsx3.init()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return data
