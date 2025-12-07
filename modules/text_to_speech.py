import tempfile
import os
import wave
import numpy

def synthesize_text(text: str) -> "numpy.ndarray":
    try:
        import pyttsx3
    except Exception:
        raise RuntimeError("pyttsx3 not installed")
    try:
        import numpy as np
    except Exception:
        raise RuntimeError("numpy not installed")

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    engine = pyttsx3.init()

    voices = engine.getProperty("voices")
    target = None
    
    neural_voices = []
    standard_voices = []
    
    for v in voices:
        name = (v.name or "").lower()
        voice_id = (v.id or "").lower()
        lang = ""
        if v.languages:
            raw = v.languages[0]
            lang = raw.lower() if isinstance(raw, str) else raw.decode("utf-8").lower()
        
        is_male = "male" in name or "male" in voice_id or "david" in name or "mark" in name or "richard" in name or "james" in name
        is_female = "female" in name or "female" in voice_id or "zira" in name or "zira" in voice_id or "eva" in name or "hazel" in name or "susan" in name or "karen" in name or "linda" in name or "aria" in name or "jenny" in name
        is_english = "en" in lang or "english" in name or "en_us" in lang or "en-gb" in lang or "en_us" in name or "en-gb" in name or "en-us" in voice_id or "en-gb" in voice_id
        
        if not is_male and is_female and is_english:
            is_neural = "neural" in name or "neural" in voice_id or "aria" in name or "jenny" in name
            voice_info = (v.id, name, voice_id, is_neural)
            if is_neural:
                neural_voices.append(voice_info)
            else:
                standard_voices.append(voice_info)
    
    if neural_voices:
        for vid, name, vid_id, _ in neural_voices:
            if "aria" in name or "jenny" in name:
                target = vid
                break
        if target is None:
            target = neural_voices[0][0]
    elif standard_voices:
        for vid, name, vid_id, _ in standard_voices:
            if "zira" in name or "zira" in vid_id:
                target = vid
                break
        if target is None:
            target = standard_voices[0][0]
    
    if target:
        engine.setProperty("voice", target)

    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()

    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        with wave.open(tmp_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)

        if sampwidth == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0

        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        if sample_rate != 16000:
            num_samples = int(len(data) * 16000 / sample_rate)
            indices = np.linspace(0, len(data) - 1, num_samples)
            data = np.interp(indices, np.arange(len(data)), data)

        return data.astype(np.float32)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
