import sounddevice as sd
import numpy as np
import queue
import time
import modules.speech_to_text as stt

samplerate = 16000
block_duration = 0.1
silence_threshold = 0.01
silence_timeout = 2.0

audio_q = queue.Queue()

def callback(indata, frames, time_, status):
    audio_q.put(indata.copy())

def on_audio_captured(data):
    print("audio captured")
    text = stt.transcribe_audio_block(data)
    print(text)

with sd.InputStream(channels=1, samplerate=samplerate, callback=callback):
    buffer = []
    recording = False
    last_voice_time = None

    while True:
        block = audio_q.get()
        volume = np.linalg.norm(block)

        if volume > silence_threshold:
            if not recording:
                recording = True
                buffer = []
            buffer.append(block)
            last_voice_time = time.time()
        else:
            if recording and last_voice_time and (time.time() - last_voice_time > silence_timeout):
                recording = False
                audio_data = np.concatenate(buffer, axis=0)
                on_audio_captured(audio_data)
                buffer = []
