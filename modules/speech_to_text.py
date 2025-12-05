from typing import Union, Optional, Tuple, List
import io

def transcribe_audio_block(audio: Union[bytes, io.BytesIO, "numpy.ndarray"],
                          model: str = "small",
                          device: str = "auto",
                          language: Optional[str] = None,
                          task: str = "transcribe",
                          return_segments: bool = False,
                          beam_size: int = 5) -> Union[str, Tuple[str, List[dict]]]:
    import numpy as np
    try:
        from faster_whisper import WhisperModel
    except:
        raise RuntimeError("faster-whisper not installed")
    use_pydub = True
    try:
        from pydub import AudioSegment
    except:
        use_pydub = False

    def ensure_numpy(x):
        if isinstance(x, np.ndarray):
            arr = x
            if arr.dtype == np.int16:
                arr = arr.astype("float32") / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype("float32") / 2147483648.0
            elif arr.dtype == np.uint8:
                arr = (arr.astype("float32") - 128) / 128.0
            elif arr.dtype == np.float64:
                arr = arr.astype("float32")
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            return arr.astype("float32")
        if isinstance(x, (bytes, io.BytesIO)):
            if not use_pydub:
                raise RuntimeError("pydub required for bytes")
            bio = io.BytesIO(x) if isinstance(x, bytes) else x
            seg = AudioSegment.from_file(bio)
            seg = seg.set_frame_rate(16000).set_channels(1)
            samples = np.array(seg.get_array_of_samples())
            if seg.sample_width == 2:
                arr = samples.astype("float32") / 32768.0
            elif seg.sample_width == 4:
                arr = samples.astype("float32") / 2147483648.0
            else:
                arr = samples.astype("float32")
                maxv = float(np.iinfo(samples.dtype).max) if np.issubdtype(samples.dtype, np.integer) else 1.0
                arr = arr / maxv
            return arr.astype("float32")
        raise TypeError("invalid audio input")

    audio_np = ensure_numpy(audio)

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            device = "cpu"

    compute_type = "float16" if device.startswith("cuda") else "int8"
    model_obj = WhisperModel(model, device=device, compute_type=compute_type)

    try:
        segments, info = model_obj.transcribe(audio_np,
                                              language=language,
                                              task=task,
                                              beam_size=beam_size)
    except:
        segments, info = model_obj.transcribe(audio=audio_np,
                                              language=language,
                                              task=task,
                                              beam_size=beam_size)

    parts = []
    segs_list = []

    for seg in segments:
        try:
            start = float(seg.start)
            end = float(seg.end)
            text = str(seg.text)
        except:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = str(seg.get("text", ""))
        parts.append(text)
        segs_list.append({"start": start, "end": end, "text": text})

    final_text = " ".join(str(p).strip() for p in parts).strip()

    if return_segments:
        return final_text, segs_list
    return final_text
