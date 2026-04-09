import subprocess
import json
import numpy as np
import sounddevice as sd
import sys
import re
import time

# Based on your uploaded starter file. :contentReference[oaicite:0]{index=0}

# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print("ERROR: ollama is not running.")
            print("Fix: open another terminal and run: ollama serve")
            sys.exit(1)
        if "qwen2.5" not in result.stdout.lower():
            print("ERROR: qwen2.5:3b not pulled.")
            print("Fix: ollama pull qwen2.5:3b")
            sys.exit(1)
        return True
    except FileNotFoundError:
        print("ERROR: ollama not installed. Get it from https://ollama.com")
        sys.exit(1)


check_ollama()

MODEL = "qwen2.5:3b"
SAMPLE_RATE = 44100


# ============================================================
# TODO #1: System prompt — tuned for reliable JSON
# ============================================================

SYNTH_PROMPT_TEMPLATE = """You generate synthesizer parameters.

Return EXACTLY ONE valid JSON object.
Return ONLY JSON.
Do not write any explanation.
Do not write markdown.
Do not use code fences.
Do not include any text before or after the JSON.
Do not include comments.
Do not include trailing commas.
Use exactly these 6 keys and no others:
base_freq, tempo, waveform, reverb, amplitude, harmonics

Allowed values:
- base_freq: integer from 80 to 800
- tempo: float from 0.3 to 3.0
- waveform: one of "sine", "triangle", "square", "sawtooth", "pulse"
- reverb: float from 0.0 to 1.0
- amplitude: float from 0.05 to 0.4
- harmonics: integer from 1 to 5

Output format example:
{{"base_freq":220,"tempo":0.8,"waveform":"sine","reverb":0.6,"amplitude":0.18,"harmonics":2}}

Mood: "{mood}"

Map moods intuitively:
- calm, peaceful, soft -> lower base_freq, slower tempo, more reverb, sine or triangle
- energetic, bright, driving -> higher base_freq, faster tempo, sawtooth or pulse
- tense, harsh, ominous -> square or pulse, medium tempo, lower reverb
- dreamy, floating, ambient -> sine or triangle, slower tempo, wetter reverb

Return ONLY the JSON object."""


def get_params_from_mood(mood):
    """Ask the LLM for synth parameters matching the mood."""
    prompt = SYNTH_PROMPT_TEMPLATE.format(mood=mood)

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True, text=True, timeout=30
        )
        raw = result.stdout.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return None

    # Try to extract JSON from the response
    # Models often add extra text or markdown — strip it out
    json_str = extract_json(raw)

    if not json_str:
        print(f"⚠️  Could not find JSON in response. Raw output:\n{raw[:200]}")
        return get_default_params()

    try:
        params = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"⚠️  Invalid JSON: {e}")
        print(f"Raw: {json_str[:200]}")
        return get_default_params()

    # Validate and clamp values
    return validate_params(params)


def extract_json(text):
    """Find the first JSON object in a string."""
    # Strip markdown code fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        return None

    return text[start:end + 1]


def validate_params(params):
    """Clamp parameters to safe ranges and fill missing fields."""
    safe = {
        "base_freq": int(np.clip(params.get("base_freq", 220), 80, 800)),
        "tempo": float(np.clip(params.get("tempo", 1.0), 0.3, 3.0)),
        "waveform": str(params.get("waveform", "sine")).lower().strip(),
        "reverb": float(np.clip(params.get("reverb", 0.3), 0.0, 1.0)),
        "amplitude": float(np.clip(params.get("amplitude", 0.2), 0.05, 0.4)),
        "harmonics": int(np.clip(params.get("harmonics", 1), 1, 5)),
    }

    # Fall back if waveform isn't valid
    if safe["waveform"] not in WAVEFORMS:
        safe["waveform"] = "sine"

    return safe


def get_default_params():
    """Safe default if LLM fails."""
    return {
        "base_freq": 220,
        "tempo": 1.0,
        "waveform": "sine",
        "reverb": 0.3,
        "amplitude": 0.2,
        "harmonics": 1,
    }


# ============================================================
# SYNTH ENGINE — pure numpy, no external libraries
# ============================================================

def gen_sine(freq, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def gen_square(freq, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sign(np.sin(2 * np.pi * freq * t))


def gen_triangle(freq, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1


def gen_sawtooth(freq, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 2 * (t * freq - np.floor(t * freq + 0.5))


# ============================================================
# TODO #2: Added new waveform — pulse
# ============================================================

def gen_pulse(freq, duration, duty=0.25, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    phase = (t * freq) % 1.0
    return np.where(phase < duty, 1.0, -1.0)


WAVEFORMS = {
    "sine": gen_sine,
    "square": gen_square,
    "triangle": gen_triangle,
    "sawtooth": gen_sawtooth,
    "pulse": gen_pulse,
}


def apply_reverb(signal, depth):
    """
    Simple delay-line reverb.
    Real reverb uses convolution with an impulse response,
    but this approximation works for our purposes.
    """
    if depth <= 0:
        return signal

    output = signal.copy()
    delays_ms = [29, 47, 73, 109]  # Prime numbers = no flutter
    decays = [0.6, 0.5, 0.4, 0.3]

    for delay_ms, decay in zip(delays_ms, decays):
        delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
        if delay_samples >= len(signal):
            continue
        delayed = np.zeros_like(signal)
        delayed[delay_samples:] = signal[:-delay_samples] * decay * depth
        output += delayed

    # Normalize to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output


def add_harmonics(signal, base_freq, num_harmonics, waveform_func, duration):
    """Layer additional harmonic frequencies on top of the base."""
    if num_harmonics <= 1:
        return signal

    output = signal.copy()
    for h in range(2, num_harmonics + 1):
        harmonic_freq = base_freq * h
        if harmonic_freq > 8000:  # Don't go above audible range too aggressively
            break
        harmonic_wave = waveform_func(harmonic_freq, duration) / (h * 1.5)
        output += harmonic_wave

    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    return output


def apply_envelope(signal, attack=0.05, release=0.3):
    """Smooth fade-in and fade-out to prevent clicks."""
    attack_samples = int(attack * SAMPLE_RATE)
    release_samples = int(release * SAMPLE_RATE)

    envelope = np.ones_like(signal)

    # Fade in
    if attack_samples > 0 and attack_samples < len(signal):
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Fade out
    if release_samples > 0 and release_samples < len(signal):
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)

    return signal * envelope


def synthesize(params, duration=8.0):
    """
    Build the audio signal from parameters.
    This is the 'parameter space → physical output' translation.
    """
    waveform_func = WAVEFORMS.get(params["waveform"], gen_sine)
    base_freq = params["base_freq"]

    # Apply tempo: rhythmic amplitude modulation
    tempo_freq = params["tempo"]  # Hz of the rhythm pulse

    # Generate base waveform
    signal = waveform_func(base_freq, duration)

    # Add harmonics for richness
    signal = add_harmonics(signal, base_freq, params["harmonics"],
                           waveform_func, duration)

    # Apply tempo modulation (LFO on amplitude)
    t = np.linspace(0, duration, len(signal), endpoint=False)
    tempo_lfo = 0.5 + 0.5 * np.sin(2 * np.pi * tempo_freq * t)
    signal = signal * tempo_lfo

    # Apply reverb
    signal = apply_reverb(signal, params["reverb"])

    # Apply amplitude
    signal = signal * params["amplitude"]

    # Smooth fade in/out
    signal = apply_envelope(signal)

    # Final clip safety
    signal = np.clip(signal, -1.0, 1.0)

    return signal.astype(np.float32)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def show_params(params):
    """Pretty print parameters."""
    print()
    print("  ┌─ Synth Parameters ─────────────────")
    print(f"  │ Waveform:  {params['waveform']}")
    print(f"  │ Frequency: {params['base_freq']} Hz")
    print(f"  │ Tempo:     {params['tempo']:.2f} Hz")
    print(f"  │ Reverb:    {'█' * int(params['reverb'] * 10):<10} {params['reverb']:.2f}")
    print(f"  │ Amplitude: {'█' * int(params['amplitude'] * 25):<10} {params['amplitude']:.2f}")
    print(f"  │ Harmonics: {params['harmonics']}")
    print("  └────────────────────────────────────")
    print()


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print()
    print("=" * 50)
    print("  🎹 MoodSynth — AI-driven ambient generator")
    print(f"  Model: {MODEL}")
    print("=" * 50)
    print()
    print("  Type a mood. The LLM converts it to synth")
    print("  parameters and plays 8 seconds of audio.")
    print()
    print("  Try: 'calm rainy night'")
    print("       'tense thriller scene'")
    print("       'peaceful forest morning'")
    print("       'glitchy alien transmission'")
    print()
    print("  Type 'quit' to exit.")
    print()

    while True:
        try:
            mood = input("Mood > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not mood:
            continue
        if mood.lower() in ("quit", "exit", "q"):
            break

        # Get parameters from LLM
        print("⏳ Asking the brain...", end="", flush=True)
        start = time.time()
        params = get_params_from_mood(mood)
        elapsed = time.time() - start
        print(f"\r                          \r", end="")

        if params is None:
            print("Failed to get parameters. Try again.")
            continue

        print(f"  ⚡ LLM responded in {elapsed:.1f}s")
        show_params(params)

        # Synthesize audio
        print("  🎵 Synthesizing audio...")
        audio = synthesize(params, duration=8.0)

        # Play it
        print("  ▶ Playing (8 seconds)...")
        try:
            sd.play(audio, samplerate=SAMPLE_RATE)
            sd.wait()
            print("  ✓ Done")
        except Exception as e:
            print(f"  ✗ Audio playback failed: {e}")
            print("  Check sounddevice.default.device — your output may be misconfigured")

        print()

    print("\nMoodSynth ended. See you tomorrow for Day 12!")


if __name__ == "__main__":
    main()
