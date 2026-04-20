import numpy as np
import matplotlib.pyplot as plt
import sys
import os

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import sounddevice as sd
    HAS_PLAYBACK = True
except ImportError:
    HAS_PLAYBACK = False


SAMPLE_RATE = 16000  # Standard for speech


# ============================================================
# SYNTHETIC ECHO GENERATION
# ============================================================

def generate_synthetic_speech(duration=3.0, sample_rate=SAMPLE_RATE):
    """Generate speech-like audio using frequency-modulated sine waves."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    signal = np.zeros_like(t)

    for burst_start in [0.2, 0.9, 1.6, 2.3]:
        burst_len = 0.5
        mask = (t >= burst_start) & (t < burst_start + burst_len)

        fundamental = np.random.uniform(100, 200)
        word = np.zeros_like(t)

        for harmonic in range(1, 5):
            word += np.sin(2 * np.pi * fundamental * harmonic * t) / harmonic

        envelope = np.zeros_like(t)
        envelope[mask] = np.sin(np.pi * (t[mask] - burst_start) / burst_len)

        signal += word * envelope

    signal = signal / np.max(np.abs(signal)) * 0.6
    return signal.astype(np.float32)


def add_synthetic_echo(signal, delay_ms=150, decay=0.5, sample_rate=SAMPLE_RATE):
    """Add a single echo to the signal."""
    delay_samples = int(sample_rate * delay_ms / 1000)
    echo = np.zeros_like(signal)

    if delay_samples < len(signal):
        echo[delay_samples:] = signal[:-delay_samples] * decay

    return signal + echo


# ============================================================
# LOAD AUDIO
# ============================================================

def load_or_generate():
    """Try to load a .wav file from current dir. Otherwise generate one."""
    wav_files = [f for f in os.listdir(".") if f.lower().endswith(".wav")]

    if wav_files and HAS_SOUNDFILE:
        path = wav_files[0]
        print(f"📂 Loading: {path}")
        try:
            data, sr = sf.read(path)

            if data.ndim > 1:
                data = data[:, 0]

            if sr != SAMPLE_RATE:
                ratio = sr / SAMPLE_RATE
                indices = (np.arange(int(len(data) / ratio)) * ratio).astype(int)
                data = data[indices]

            print(f"   Loaded {len(data) / SAMPLE_RATE:.1f}s of audio")
            return data.astype(np.float32), True

        except Exception as e:
            print(f"   Load failed: {e}")

    print("📡 No .wav file found — generating synthetic speech")
    clean = generate_synthetic_speech()
    return clean, False


# ============================================================
# TODO #1: FIR filter order
# ============================================================
# Echo delay = 150 ms
# 150 ms at 16 kHz = 2400 samples
# So the filter order must be at least around 2400
FILTER_ORDER = 2500


# ============================================================
# TODO #2: LMS learning rate (mu)
# ============================================================
# Chosen as a stable value in the recommended range
LEARNING_RATE = 0.02


# ============================================================
# LMS ADAPTIVE FILTER
# ============================================================

def lms_filter(reference, mixed, filter_order, mu):
    """
    Run LMS adaptive filter.

    Args:
        reference: the clean signal
        mixed: signal containing echo
        filter_order: number of FIR taps
        mu: learning rate

    Returns:
        error_signal: mixed - predicted_echo
        coefficients: final learned FIR taps
    """
    N = len(mixed)

    w = np.zeros(filter_order, dtype=np.float32)
    error = np.zeros(N, dtype=np.float32)
    ref_buffer = np.zeros(filter_order, dtype=np.float32)

    for n in range(N):
        ref_buffer[1:] = ref_buffer[:-1]
        ref_buffer[0] = reference[n]

        predicted_echo = np.dot(w, ref_buffer)

        e = mixed[n] - predicted_echo
        error[n] = e

        norm = np.dot(ref_buffer, ref_buffer) + 1e-6
        w = w + (mu / norm) * e * ref_buffer

    return error, w


# ============================================================
# TODO #3: Understand the coefficients
# ============================================================
# The learned coefficients represent the room's impulse response.
# Each coefficient tells how much of the past signal returns as echo
# after a certain delay.
#
# Example:
# - A large spike near sample 2400 means there is a strong echo
#   arriving about 150 ms later.
# - More spikes mean more reflections from walls or surfaces.
#
# So the coefficients are basically a map of how the room "colors"
# and delays the sound.


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(clean, echoed, cleaned, coefficients, sample_rate):
    """Side-by-side comparison plots."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 9))
    fig.suptitle("EchoKiller — Day 16", fontsize=14, fontweight='bold')

    t = np.arange(len(clean)) / sample_rate

    axes[0].plot(t, clean, color='#43a047', linewidth=0.8)
    axes[0].set_title("1. Clean Reference (input to room)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, echoed, color='#e53935', linewidth=0.8)
    axes[1].set_title("2. With Echo (mic input)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, cleaned, color='#1e88e5', linewidth=0.8)
    axes[2].set_title("3. After EchoKiller (LMS output)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    tap_times = np.arange(len(coefficients)) / sample_rate * 1000
    axes[3].stem(
        tap_times[::5],
        coefficients[::5],
        basefmt=" ",
        linefmt='#ff6f00',
        markerfmt='o'
    )
    axes[3].set_title(
        f"4. Learned FIR Coefficients — the room's impulse response ({len(coefficients)} taps)"
    )
    axes[3].set_xlabel("Delay (ms)")
    axes[3].set_ylabel("Amplitude")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 50)
    print("  🔇 EchoKiller — Adaptive FIR Echo Cancellation")
    print("=" * 50)
    print()

    clean, from_file = load_or_generate()

    if not from_file:
        print("📢 Adding synthetic echo (150ms delay, 50% decay)")
        echoed = add_synthetic_echo(clean, delay_ms=150, decay=0.5)
    else:
        print("📢 Adding extra synthetic echo for demonstration")
        echoed = add_synthetic_echo(clean, delay_ms=150, decay=0.5)

    print(f"\n⚙️  LMS Parameters:")
    print(f"   Filter order: {FILTER_ORDER} taps")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Audio length: {len(clean) / SAMPLE_RATE:.1f}s ({len(clean)} samples)")

    print("\n🧠 Running LMS adaptive filter...")
    import time
    start = time.time()
    cleaned, coefficients = lms_filter(clean, echoed, FILTER_ORDER, LEARNING_RATE)
    elapsed = time.time() - start
    print(f"   Done in {elapsed:.1f}s")

    echo_energy = np.mean(echoed ** 2)
    cleaned_energy = np.mean(cleaned ** 2)
    clean_energy = np.mean(clean ** 2)

    print(f"\n📊 Energy analysis:")
    print(f"   Clean signal:   {clean_energy:.6f}")
    print(f"   With echo:      {echo_energy:.6f}")
    print(f"   After filter:   {cleaned_energy:.6f}")

    reduction_db = 10 * np.log10(echo_energy / max(cleaned_energy, 1e-9))
    print(f"   Noise reduction: {reduction_db:.1f} dB")

    if HAS_PLAYBACK:
        print("\n🔊 Playback order: clean → echoed → cleaned")
        print("   (press Ctrl+C to skip)")
        try:
            print("   Playing CLEAN...")
            sd.play(clean, SAMPLE_RATE)
            sd.wait()

            print("   Playing ECHOED...")
            sd.play(echoed, SAMPLE_RATE)
            sd.wait()

            print("   Playing CLEANED...")
            sd.play(cleaned, SAMPLE_RATE)
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print("   Playback skipped.")

    if HAS_SOUNDFILE:
        sf.write("cleaned_output.wav", cleaned, SAMPLE_RATE)
        sf.write("echoed_input.wav", echoed, SAMPLE_RATE)
        print("\n💾 Saved: cleaned_output.wav, echoed_input.wav")

    print("\n📈 Displaying waveforms and filter coefficients...")
    plot_results(clean, echoed, cleaned, coefficients, SAMPLE_RATE)

    print("\nSee you tomorrow for Day 17!")


if __name__ == "__main__":
    main()
