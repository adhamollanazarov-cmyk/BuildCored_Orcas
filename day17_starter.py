"""
BUILDCORED ORCAS — Day 17: PWMSimulator
Completed version:
- TODO #1 done: added a second PWM channel at a different frequency
- TODO #2 done: clarified why duty cycle determines average voltage

Run: python day17_starter.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle

# ============================================================
# PWM PARAMETERS
# ============================================================

VCC = 3.3                # Supply voltage
PWM1_FREQ = 1000         # Channel 1 frequency in Hz
PWM2_FREQ = 500          # Channel 2 frequency in Hz
DISPLAY_TIME_MS = 5      # Show 5 ms total on screen
SAMPLE_POINTS = 2000     # Resolution of waveform plot

# Current duty cycles
current_duty_1 = 50.0
current_duty_2 = 30.0


# ============================================================
# PWM WAVEFORM GENERATION
# ============================================================

def generate_pwm_wave(duty_percent, freq, display_time_ms=DISPLAY_TIME_MS, points=SAMPLE_POINTS):
    """
    Generate a PWM square wave for a fixed display window.
    Returns (time_array_seconds, voltage_array).

    Duty cycle = time HIGH / total period
    """
    total_time = display_time_ms / 1000.0
    t = np.linspace(0, total_time, points)

    period = 1.0 / freq
    position_in_period = (t % period) / period
    duty_fraction = duty_percent / 100.0
    voltage = np.where(position_in_period < duty_fraction, VCC, 0.0)

    return t, voltage


def compute_average_voltage(duty_percent):
    """
    Why does duty cycle set average voltage?

    Average voltage over one full PWM period is:

        average = (V_high * time_high + V_low * time_low) / total_time

    For PWM:
        V_high = VCC
        V_low = 0
        duty_fraction = time_high / total_time

    So:

        average = (VCC * time_high + 0 * time_low) / total_time
                = VCC * (time_high / total_time)
                = VCC * duty_fraction

    Example:
        25% duty on 3.3 V PWM gives:
        average = 3.3 * 0.25 = 0.825 V

    Important:
    The pin is never outputting a "half voltage" directly.
    It is only rapidly switching FULL ON and FULL OFF.
    The average over time is what loads like LEDs and motors respond to.
    """
    return VCC * (duty_percent / 100.0)


def led_style_from_duty(duty_percent):
    """Return a color and alpha values for LED visualization."""
    brightness = duty_percent / 100.0

    if brightness == 0:
        color = '#5d4037'   # dark brown-ish
        state = "OFF"
    elif brightness < 0.3:
        color = '#ef6c00'   # dim orange
        state = "DIM"
    elif brightness < 0.7:
        color = '#ffa000'   # amber
        state = "MEDIUM"
    else:
        color = '#ffeb3b'   # bright yellow
        state = "BRIGHT"

    circle_alpha = 0.12 + brightness * 0.88
    glow_alpha = brightness * 0.40

    return color, circle_alpha, glow_alpha, state


# ============================================================
# PLOT SETUP
# ============================================================

fig = plt.figure(figsize=(13, 8))
fig.suptitle("PWMSimulator — Day 17 (Dual Channel)", fontsize=14, fontweight='bold')

# Main waveform plot
ax_wave = plt.axes([0.08, 0.44, 0.62, 0.46])
ax_wave.set_xlim(0, DISPLAY_TIME_MS)
ax_wave.set_ylim(-0.3, VCC + 0.5)
ax_wave.set_xlabel("Time (ms)")
ax_wave.set_ylabel("Voltage (V)")
ax_wave.set_title("PWM Waveforms")
ax_wave.grid(True, alpha=0.3)

# Wave lines
wave1_line, = ax_wave.plot([], [], linewidth=2, label=f'Channel 1 ({PWM1_FREQ} Hz)')
wave2_line, = ax_wave.plot([], [], linewidth=2, label=f'Channel 2 ({PWM2_FREQ} Hz)')

# Average voltage lines
avg1_line = ax_wave.axhline(y=VCC / 2, linestyle='--', linewidth=1.5, label='Ch1 Average')
avg2_line = ax_wave.axhline(y=VCC / 3, linestyle=':', linewidth=1.5, label='Ch2 Average')

ax_wave.legend(loc='upper right')

# LED area
ax_led = plt.axes([0.74, 0.46, 0.22, 0.40])
ax_led.set_xlim(-2.2, 2.2)
ax_led.set_ylim(-2.0, 2.0)
ax_led.set_aspect('equal')
ax_led.axis('off')
ax_led.set_title("Virtual LEDs", fontweight='bold')

# Channel 1 LED
led1_glow = Circle((-0.9, 0), 1.1, color='#ffeb3b', alpha=0.2)
led1_circle = Circle((-0.9, 0), 0.8, color='#ffeb3b', alpha=0.5)
ax_led.add_patch(led1_glow)
ax_led.add_patch(led1_circle)
ax_led.add_patch(plt.Rectangle((-1.35, -1.25), 0.9, 0.22, color='#424242'))
led1_label = ax_led.text(-0.9, 1.35, "CH1", ha='center', fontsize=10, fontweight='bold')
led1_info = ax_led.text(-0.9, -1.55, "", ha='center', fontsize=9, fontweight='bold')

# Channel 2 LED
led2_glow = Circle((0.9, 0), 1.1, color='#90caf9', alpha=0.2)
led2_circle = Circle((0.9, 0), 0.8, color='#90caf9', alpha=0.5)
ax_led.add_patch(led2_glow)
ax_led.add_patch(led2_circle)
ax_led.add_patch(plt.Rectangle((0.45, -1.25), 0.9, 0.22, color='#424242'))
led2_label = ax_led.text(0.9, 1.35, "CH2", ha='center', fontsize=10, fontweight='bold')
led2_info = ax_led.text(0.9, -1.55, "", ha='center', fontsize=9, fontweight='bold')

# Sliders
slider1_ax = plt.axes([0.14, 0.25, 0.70, 0.035])
duty1_slider = Slider(
    ax=slider1_ax,
    label="Duty Cycle CH1 (%)",
    valmin=0,
    valmax=100,
    valinit=current_duty_1,
    valstep=1,
    color='#0f7173',
)

slider2_ax = plt.axes([0.14, 0.18, 0.70, 0.035])
duty2_slider = Slider(
    ax=slider2_ax,
    label="Duty Cycle CH2 (%)",
    valmin=0,
    valmax=100,
    valinit=current_duty_2,
    valstep=1,
    color='#8e24aa',
)

# Stats panel
stats_ax = plt.axes([0.08, 0.03, 0.88, 0.11])
stats_ax.axis('off')
stats_text = stats_ax.text(
    0.5, 0.5, "",
    ha='center', va='center',
    fontsize=10,
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='#f0f0f0', pad=1),
    transform=stats_ax.transAxes
)


# ============================================================
# UPDATE LOGIC
# ============================================================

def update_display(_=None):
    """Refresh everything based on both duty cycle sliders."""
    global current_duty_1, current_duty_2

    current_duty_1 = duty1_slider.val
    current_duty_2 = duty2_slider.val

    # Generate waveforms
    t1, v1 = generate_pwm_wave(current_duty_1, PWM1_FREQ)
    t2, v2 = generate_pwm_wave(current_duty_2, PWM2_FREQ)

    # Update waveform lines
    wave1_line.set_data(t1 * 1000, v1)
    wave2_line.set_data(t2 * 1000, v2)

    # Update average lines
    avg1 = compute_average_voltage(current_duty_1)
    avg2 = compute_average_voltage(current_duty_2)
    avg1_line.set_ydata([avg1, avg1])
    avg2_line.set_ydata([avg2, avg2])

    # Update LED 1
    color1, circle_alpha1, glow_alpha1, state1 = led_style_from_duty(current_duty_1)
    led1_circle.set_color(color1)
    led1_circle.set_alpha(circle_alpha1)
    led1_glow.set_color(color1)
    led1_glow.set_alpha(glow_alpha1)
    led1_info.set_text(f"{state1}\n{current_duty_1:.0f}%")

    # Update LED 2
    color2, circle_alpha2, glow_alpha2, state2 = led_style_from_duty(current_duty_2)
    led2_circle.set_color(color2)
    led2_circle.set_alpha(circle_alpha2)
    led2_glow.set_color(color2)
    led2_glow.set_alpha(glow_alpha2)
    led2_info.set_text(f"{state2}\n{current_duty_2:.0f}%")

    # Timing stats
    period1_us = 1e6 / PWM1_FREQ
    high1_us = period1_us * (current_duty_1 / 100.0)
    low1_us = period1_us - high1_us

    period2_us = 1e6 / PWM2_FREQ
    high2_us = period2_us * (current_duty_2 / 100.0)
    low2_us = period2_us - high2_us

    stats = (
        f"CH1: {PWM1_FREQ:4d} Hz | Period: {period1_us:7.0f} µs | HIGH: {high1_us:7.0f} µs | LOW: {low1_us:7.0f} µs | Avg V: {avg1:4.2f} V\n"
        f"CH2: {PWM2_FREQ:4d} Hz | Period: {period2_us:7.0f} µs | HIGH: {high2_us:7.0f} µs | LOW: {low2_us:7.0f} µs | Avg V: {avg2:4.2f} V\n"
        f"Why duty cycle = average voltage: average = VCC × duty_fraction   |   VCC = {VCC} V"
    )
    stats_text.set_text(stats)

    fig.canvas.draw_idle()


# Hook sliders
duty1_slider.on_changed(update_display)
duty2_slider.on_changed(update_display)

# Initial render
update_display()


# ============================================================
# RUN
# ============================================================

print("\n" + "=" * 60)
print("  💡 PWMSimulator — Day 17 (Dual Channel)")
print("=" * 60)
print()
print("  Drag either slider to change the duty cycle.")
print("  Channel 1 runs at 1000 Hz.")
print("  Channel 2 runs at 500 Hz.")
print()
print("  Watch how:")
print("   - waveforms differ because frequencies differ")
print("   - LEDs brighten as duty cycle increases")
print("   - average voltage follows: Vavg = VCC × duty_fraction")
print()
print("  Close the plot window to quit.")
print()

plt.show()

print("\nPWMSimulator ended. See you tomorrow for Day 18!")
