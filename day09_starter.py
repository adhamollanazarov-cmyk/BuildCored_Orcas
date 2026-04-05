"""
BUILDCORED ORCAS — Day 09: WhisperDesk
========================================
Local speech-to-text. Speak into your mic, see the
text appear in your terminal. No cloud. No API keys.

Hardware concept: On-Device DSP + Latency Budgeting
Audio chunk size sets your latency budget:
  - Small chunks (2s): fast response, less accurate
  - Large chunks (5s): slow response, more accurate
This is the SAME tradeoff firmware engineers make when
sizing audio buffers on embedded devices.

YOUR TASK:
1. Tune the chunk size for latency vs accuracy (TODO #1)
2. Understand the audio pipeline stages (TODO #2)
3. Run it: python day09_starter.py
4. Push to GitHub before midnight

PREREQUISITES:
Either:
  a) pip install faster-whisper   (recommended, ~1 GB model download)
  b) ollama pull qwen2.5:3b      (fallback, uses ollama for transcription)

CONTROLS:
- Speak into your microphone
- Text appears after each chunk is processed
- Press Ctrl+C to quit
"""

import pyaudio
import numpy as np
import wave
import tempfile
import os
import sys
import time
import subprocess

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

SILENCE_THRESHOLD = 500

BACKEND = None


# ================================
# BACKEND SETUP
# ================================

def setup_faster_whisper():
    global BACKEND
    try:
        from faster_whisper import WhisperModel
        print("Loading faster-whisper model...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        BACKEND = "faster-whisper"
        print("✓ faster-whisper ready")
        return model
    except:
        return None


def setup_ollama():
    global BACKEND
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True)
        if result.returncode == 0:
            BACKEND = "ollama"
            print("✓ Using ollama fallback")
            return True
    except:
        return False


whisper_model = setup_faster_whisper()

if whisper_model is None:
    if not setup_ollama():
        print("No backend available")
        sys.exit(1)


# ================================
# SMART LATENCY TUNING (TODO #1)
# ================================
if BACKEND == "faster-whisper":
    RECORD_SECONDS = 3
else:
    RECORD_SECONDS = 5


# ================================
# TRANSCRIPTION
# ================================

def transcribe(audio_file_path):
    if BACKEND == "faster-whisper":
        segments, _ = whisper_model.transcribe(
            audio_file_path,
            beam_size=1,
            language="en",
            vad_filter=True,
        )
        return " ".join(s.text for s in segments).strip()

    else:
        return "[Speech detected - ollama fallback]"


# ================================
# AUDIO
# ================================

def is_silent(audio_data):
    samples = np.frombuffer(audio_data, dtype=np.int16)
    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    return rms < SILENCE_THRESHOLD


def record_chunk():
    frames = []
    num_frames = int(RATE / CHUNK * RECORD_SECONDS)

    for _ in range(num_frames):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    audio_data = b''.join(frames)

    if is_silent(audio_data):
        return None

    return audio_data


def save_audio(audio_data):
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
    return temp.name


# ================================
# MIC INIT
# ================================

pa = pyaudio.PyAudio()

device_index = None
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        device_index = i
        print(f"Using mic: {info['name']}")
        break

stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=CHUNK,
)

print("\n🎙️ WhisperDesk STARTED")
print(f"Backend: {BACKEND}")
print(f"Chunk: {RECORD_SECONDS}s\n")


# ================================
# MAIN LOOP
# ================================

try:
    while True:
        print(f"\n🔴 Listening ({RECORD_SECONDS}s)...", end="")

        audio = record_chunk()

        if audio is None:
            print("\r⚪ Silence skipped")
            continue

        path = save_audio(audio)

        print("\r⏳ Transcribing...", end="")

        start = time.time()
        text = transcribe(path)
        inference_time = time.time() - start

        os.unlink(path)

        if text:
            total_latency = RECORD_SECONDS + inference_time

            print("\r📝", text)

            # ================================
            # PIPELINE DEBUG (TODO #2)
            # ================================
            print("🧠 PIPELINE:")
            print(f"   Capture: continuous")
            print(f"   Buffer: {RECORD_SECONDS:.1f}s")
            print(f"   Inference: {inference_time:.1f}s")
            print(f"   Total latency: {total_latency:.1f}s")

except KeyboardInterrupt:
    pass

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("\nStopped.")
