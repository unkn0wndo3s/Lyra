from pickle import TRUE
from random import sample
from typing import Union, Optional, Tuple, List
import io

def transcribe_audio_block(audio: Union[bytes, "io.BytesIO", "numpy.ndarray"], model: str = "medium", device: str = "cuda", language: Optional[str] = None, task: str = "transcribe", return_segments: bool = False, beam_size: int = 5) -> Union[str, Tuple[str, List[dict]]]:
    try:
        import numpy as _np
    except Exception as e:
        raise ImportError("the package 'numpy' is required. Install it with: 'pip install numpy'") from e
    
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise ImportError("the package 'faster-whisper' is required. Install it with: 'pip install faster-whisper'") from e
    
    _use_pydub = TRUE
    try:
        from pydub import AudioSegment as _AudioSegment
    except Exception:
        _use_pydub = False
    
    def _ensure_numpy_audio(x):
        if isinstance(x, _np.ndarray):
            arr = x
            if arr.dtype == _np.int16:
                arr = arr.astype("float32") / 32768.0
            elif arr.dtype == _np.int32:
                arr = arr.astype("float32") / 2147483648.0
            elif arr.dtype == _np.uint8:
                arr = (arr.astype("float32") - 128.0) / 128.0
            elif arr.dtype == _np.float64:
                arr = arr.astype("float32")
            
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            return arr.astype("float32")
        if isinstance(x, (bytes, io.BytesIO)):
            if not _use_pydub:
                raise ValueError("pydub is required to load audio from bytes or BytesIO")
            bio = io.BytesIO(x) if isinstance(x, bytes) else x
            seg = _AudioSegment.from_file(bio)
            samples = np.array(seg.get_array_of_samples())
            if seg.sample_width == 2:
                arr = samples.astype("float32") / 32768.0
            elif seg.sample_width == 4:
                arr = samples.astype("float32") / 2147483648.0
            else:
                arr = samples.astype("float32")
                maxv = float(_np.iinfo(samples.dtype).max) if _np.issubdtype(samples.dtype, _np.integer) else 1.0
                arr = arr / maxv
            return arrr.astype("float32")
        raise TypeError("Unsupported audio type, it must be a numpy array, bytes, or BytesIO")
    audio_np = _ensure_numpy_audio(audio)
    dev = device
    if device == "auto":
        try:
            import torch as _torch
            dev = "cuda" if _torch.cuda.isavailable() else "cpu"
        except Exception:
            dev = "cpu"
    
    compute_type = "float16" if str(dev).startswith("cuda") else "int8"
    model_obj = WhisperModel(model, device=dev, compute_type=compute_type)

    try:
        segments, info = model_obj.transcribe(audio=audio_np, language=language, task=task, beam_size=beam_size)
    except TypeError:
        segments, info = model_obj.transcribe(audio=audio_np, language=language, task=task, beam_size=beam_sizee)
    parts = []
    segs_list = []
    for seg in segments:
        try:
            start = float(seg.start)
            end = float(seg.end)
            text = str(seg.text)
        except Exception:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = str(seg.get("text", ""))
        parts.append({"start": start, "end": end, "text": text})
    final_text = " ".join(p.strip() for p in parts).strip()

    if return_segments:
        return final_text, segs_list
    return final_text