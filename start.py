import sounddevice as sd
import numpy as np
import queue
import time
import modules.speech_to_text as stt
import modules.AI as ai

samplerate = 16000
block_duration = 0.1
silence_threshold = 0.01
silence_timeout = 2.0

audio_q = queue.Queue()

def callback(indata, frames, time_, status):
    audio_q.put(indata.copy())

def on_audio_captured(data):
    print("transcribing audio...")
    text = stt.transcribe_audio_block(data)
    if text == "":
        print("no text detected")
        return
    print("sending audio to AI...")
    response = ai.send_history([{"role": "user", "content": text}], "")
    print("AI response:", response)

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
