import os
import tempfile
import soundfile as sf
import sounddevice as sd

def speak_fallback(text: str):
    try:
        import pyttsx3
    except Exception:
        raise RuntimeError("pyttsx3 missing")
    t = pyttsx3.init()
    path = os.path.join(tempfile.gettempdir(), "tts_fallback.wav")
    t.save_to_file(text, path)
    t.runAndWait()
    data, sr = sf.read(path, dtype="float32")
    sd.play(data, sr); sd.wait()
    return path

def speak_vibe(text: str):
    try:
        from transformers import pipeline
        tts = pipeline("text-to-speech", model="microsoft/VibeVoice-1.5B", trust_remote_code=True)
        out = tts(text)
        wav = out["wav"]
        sr = out.get("sample_rate", 24000)
        # tensor -> numpy
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        # channels first -> transpose
        if getattr(wav, "ndim", 1) == 2 and wav.shape[0] > wav.shape[1]:
            wav = wav.T
        path = os.path.join(tempfile.gettempdir(), "vibe.wav")
        sf.write(path, wav, sr)
        data, sr = sf.read(path, dtype="float32")
        sd.play(data, sr); sd.wait()
        return path
    except Exception as e:
        # fallback sans chichi
        print("fallback")
        return speak_fallback(text)

# call
if __name__ == "__main__":
    speak_vibe("Hello, this is a test! AAAAAAAAAAAAAH")
