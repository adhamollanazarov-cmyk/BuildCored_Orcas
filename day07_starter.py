"Day 07: KeyboardOscilloscope"

import numpy as np
import sounddevice as sd
import pygame
import sys
import threading

SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
AMPLITUDE = 0.15       

active_frequencies = set()
lock = threading.Lock()


def audio_callback(outdata, frames, time_info, status):
    """
    Called by sounddevice to fill the audio buffer.
    Generates sine waves for all active frequencies
    and adds them together (superposition).
    """
    t = np.arange(frames) / SAMPLE_RATE

    if not hasattr(audio_callback, 'phase'):
        audio_callback.phase = 0.0

    signal = np.zeros(frames, dtype=np.float32)

    with lock:
        freqs = list(active_frequencies)

    for freq in freqs:
        wave = AMPLITUDE * np.sin(2 * np.pi * freq * t + audio_callback.phase)
        signal += wave

    signal = np.clip(signal, -0.8, 0.8)

    audio_callback.phase += 2 * np.pi * frames / SAMPLE_RATE

    outdata[:, 0] = signal


audio_stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BUFFER_SIZE,
    channels=1,
    dtype='float32',
    callback=audio_callback,
)
audio_stream.start()


# ============================================================
# TODO #1: Key-to-Frequency Mapping
# ============================================================
# Each keyboard key maps to a musical frequency.
# These are piano note frequencies (A4 = 440 Hz).
#
# The bottom row (Z-M) is one octave.
# The middle row (A-L) is the next octave.
# The top row (Q-P) is the highest octave.
#
# ADD MORE KEYS to fill out the keyboard!
# Musical note frequencies: https://pages.mtu.edu/~suits/notefreqs.html
#
# pygame key constants: pygame.K_a, pygame.K_b, etc.
#

KEY_TO_FREQ = {
    # Bottom row — low octave (C4–B4)
    pygame.K_z: 261.63,   # C4
    pygame.K_x: 293.66,   # D4
    pygame.K_c: 329.63,   # E4
    pygame.K_v: 349.23,   # F4
    pygame.K_b: 392.00,   # G4
    pygame.K_n: 440.00,   # A4
    pygame.K_m: 493.88,   # B4

    # Middle row — mid octave (C5–B5)
    pygame.K_a: 523.25,   # C5
    pygame.K_s: 587.33,   # D5
    pygame.K_d: 659.25,   # E5
    pygame.K_f: 698.46,   # F5
    pygame.K_g: 783.99,   # G5
    pygame.K_h: 880.00,   # A5
    pygame.K_j: 987.77,   # B5

    # Top row — high octave (C6–B6)
    pygame.K_q: 1046.50,  # C6
    pygame.K_w: 1174.66,  # D6
    pygame.K_e: 1318.51,  # E6
    pygame.K_r: 1396.91,  # F6
    pygame.K_t: 1567.98,  # G6
    pygame.K_y: 1760.00,  # A6
    pygame.K_u: 1975.53,  # B6
}

# Reverse lookup for display
FREQ_TO_NOTE = {
    261.63: "C4", 293.66: "D4", 329.63: "E4", 349.23: "F4",
    392.00: "G4", 440.00: "A4", 493.88: "B4",
    523.25: "C5", 587.33: "D5", 659.25: "E5", 698.46: "F5",
    783.99: "G5", 880.00: "A5", 987.77: "B5",
    1046.50: "C6", 1174.66: "D6", 1318.51: "E6", 1396.91: "F6",
    1567.98: "G6", 1760.00: "A6", 1975.53: "B6",
}



pygame.init()
WIDTH = 900
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("KeyboardOscilloscope — Day 07")
clock = pygame.time.Clock()

# Colors
BG_COLOR = (10, 10, 18)
GRID_COLOR = (30, 30, 45)
TEXT_COLOR = (180, 180, 200)
WAVE_COLORS = [
    (79, 195, 247),   # Light blue
    (129, 199, 132),   # Green
    (255, 183, 77),    # Orange
    (240, 98, 146),    # Pink
    (186, 104, 200),   # Purple
    (255, 241, 118),   # Yellow
    (128, 222, 234),   # Cyan
    (255, 138, 101),   # Coral
    (174, 213, 129),   # Light green
    (149, 117, 205),   # Violet
    (255, 213, 79),    # Gold
    (100, 181, 246),   # Blue
    (239, 154, 154),   # Light red
    (129, 212, 250),   # Sky blue
]
SUM_COLOR = (255, 255, 255)
ZERO_LINE_COLOR = (50, 50, 70)

FONT = pygame.font.SysFont("monospace", 14)
FONT_LARGE = pygame.font.SysFont("monospace", 20, bold=True)


def generate_waveform(freq, num_points, cycles=3):
    """Generate one sine wave for display."""
    t = np.linspace(0, cycles / freq, num_points)
    return np.sin(2 * np.pi * freq * t)


# ============================================================
# TODO #2: Understand superposition
# ============================================================
# The draw_oscilloscope function below draws:
#   1. Each individual sine wave (thin colored lines)
#   2. The SUM of all waves (thick white line)
#
# The sum IS superposition. When two waves are in phase
# (peaks align), they add up (constructive interference).
# When they're out of phase (peak meets trough), they
# cancel out (destructive interference).
#
# Try these experiments:
# - Hold Z (C4, 261 Hz) and A (C5, 523 Hz) — octave apart
#   Notice the sum wave has a clear pattern
# - Hold Z and X and C — a chord
#   The sum wave gets complex but still periodic
# - Hold many keys at once — the sum approaches noise
#

def draw_oscilloscope(surface, frequencies):
    """Draw individual waveforms and their superposition."""

    # Waveform area
    wave_top = 60
    wave_bottom = HEIGHT - 100
    wave_height = wave_bottom - wave_top
    wave_mid = wave_top + wave_height // 2
    num_points = WIDTH - 40
    margin_left = 20

    # Draw zero line
    pygame.draw.line(surface, ZERO_LINE_COLOR,
                     (margin_left, wave_mid), (WIDTH - 20, wave_mid), 1)

    # Draw grid lines
    for i in range(1, 4):
        y_up = wave_mid - int(wave_height * 0.25 * i)
        y_down = wave_mid + int(wave_height * 0.25 * i)
        pygame.draw.line(surface, GRID_COLOR,
                         (margin_left, y_up), (WIDTH - 20, y_up), 1)
        pygame.draw.line(surface, GRID_COLOR,
                         (margin_left, y_down), (WIDTH - 20, y_down), 1)

    if not frequencies:
        # No active frequencies — show flat line
        text = FONT.render("Press keys Z-M (low) or A-J (high) to play tones",
                           True, TEXT_COLOR)
        surface.blit(text, (margin_left, wave_mid - 10))
        return

    # Generate and draw individual waveforms
    sum_wave = np.zeros(num_points)
    scale = wave_height * 0.2  # Scale factor for each wave

    for i, freq in enumerate(sorted(frequencies)):
        wave = generate_waveform(freq, num_points)
        sum_wave += wave

        # Draw individual wave (thin)
        color = WAVE_COLORS[i % len(WAVE_COLORS)]
        points = []
        for x in range(num_points):
            y = int(wave_mid - wave[x] * scale)
            points.append((margin_left + x, y))

        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 1)

        # Label
        note = FREQ_TO_NOTE.get(freq, f"{freq:.0f}")
        label = FONT.render(f"{note} ({freq:.0f} Hz)", True, color)
        surface.blit(label, (WIDTH - 180, wave_top + i * 20))

    # Draw superposition (thick white line)
    # Normalize so it doesn't clip
    max_val = max(abs(sum_wave.max()), abs(sum_wave.min()), 1)
    sum_normalized = sum_wave / max_val

    sum_scale = wave_height * 0.35
    points = []
    for x in range(num_points):
        y = int(wave_mid - sum_normalized[x] * sum_scale)
        points.append((margin_left + x, y))

    if len(points) > 1:
        pygame.draw.lines(surface, SUM_COLOR, False, points, 2)

    # Label for sum
    label = FONT.render(f"SUM ({len(frequencies)} waves)", True, SUM_COLOR)
    surface.blit(label, (WIDTH - 180, wave_top + len(frequencies) * 20 + 10))



def draw_keyboard_hint(surface):
    """Draw a mini keyboard showing which keys are mapped."""
    key_size = 32
    gap = 4

    rows = [
        ("High:", "QWERTYU", ["C6","D6","E6","F6","G6","A6","B6"], (240, 98, 146),  HEIGHT - 120),
        ("Mid: ", "ASDFGHJ", ["C5","D5","E5","F5","G5","A5","B5"], (129, 199, 132), HEIGHT - 80),
        ("Low: ", "ZXCVBNM", ["C4","D4","E4","F4","G4","A4","B4"], (79,  195, 247), HEIGHT - 40),
    ]

    for row_label, keys, notes, active_color, y_start in rows:
        label = FONT.render(row_label, True, TEXT_COLOR)
        surface.blit(label, (10, y_start + 8))

        for i, (key_char, note) in enumerate(zip(keys, notes)):
            x = 60 + i * (key_size + gap)
            pg_key = getattr(pygame, f"K_{key_char.lower()}")
            freq = KEY_TO_FREQ.get(pg_key)
            is_active = freq in active_frequencies if freq else False

            color = active_color if is_active else (40, 40, 55)
            border = (100, 100, 120)
            pygame.draw.rect(surface, color, (x, y_start, key_size, key_size), border_radius=4)
            pygame.draw.rect(surface, border, (x, y_start, key_size, key_size), 1, border_radius=4)

            char_text = FONT.render(key_char, True, (255, 255, 255) if is_active else TEXT_COLOR)
            surface.blit(char_text, (x + 10, y_start + 2))
            note_text = FONT.render(note, True, (200, 200, 200) if is_active else (80, 80, 100))
            surface.blit(note_text, (x + 2, y_start + 18))

# ============================================================
# MAIN LOOP                          ← next section starts here


# ============================================================
# MAIN LOOP
# ============================================================
print("\nKeyboardOscilloscope is running!")
print("Click the window, then press keys to play tones.")
print("Bottom row (Z-M): C4 to B4")
print("Middle row (A-J): C5 to B5")
print("Hold multiple keys for chords.")
print("ESC or close window to quit.\n")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key in KEY_TO_FREQ:
                freq = KEY_TO_FREQ[event.key]
                with lock:
                    active_frequencies.add(freq)
                note = FREQ_TO_NOTE.get(freq, f"{freq:.0f}")
                print(f"♪ {note} ({freq:.0f} Hz)")

        elif event.type == pygame.KEYUP:
            if event.key in KEY_TO_FREQ:
                freq = KEY_TO_FREQ[event.key]
                with lock:
                    active_frequencies.discard(freq)

    # Draw
    screen.fill(BG_COLOR)

    # Title
    title = FONT_LARGE.render("KeyboardOscilloscope", True, TEXT_COLOR)
    screen.blit(title, (20, 15))

    active_count = len(active_frequencies)
    if active_count > 0:
        info = FONT.render(f"{active_count} active tone{'s' if active_count != 1 else ''}",
                           True, (79, 195, 247))
    else:
        info = FONT.render("No active tones — press a key", True, (80, 80, 100))
    screen.blit(info, (280, 20))

    # Draw oscilloscope
    with lock:
        freqs_snapshot = set(active_frequencies)
    draw_oscilloscope(screen, freqs_snapshot)

    # Draw keyboard hint
    draw_keyboard_hint(screen)

    pygame.display.flip()
    clock.tick(30)

# Cleanup
audio_stream.stop()
audio_stream.close()
pygame.quit()
print("\nKeyboardOscilloscope ended.")
print("Week 1 complete! See you Monday for Week 2! 🎉")
