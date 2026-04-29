# Save as: day21_starter.py

import socket
import struct
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections
import sys

HOST = "127.0.0.1"
PORT = 5005
SEND_RATE_HZ = 200
BUFFER_SIZE = 512
PACKET_FORMAT = "!Hfd"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

recv_buffer = collections.deque(maxlen=BUFFER_SIZE)
recv_lock = threading.Lock()

stats = {
    "sent": 0,
    "received": 0,
    "dropped": 0,
    "loss_pct": 0.0,
    "last_seq": -1,
    "out_of_order": 0,
}
stats_lock = threading.Lock()

loss_enabled = threading.Event()
noise_enabled = threading.Event()
running = True


def sender_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    t_start = time.time()
    interval = 1.0 / SEND_RATE_HZ

    while running:
        t = time.time() - t_start

        value = np.sin(2 * np.pi * 2.0 * t)
        if noise_enabled.is_set():
            value += np.random.normal(0, 0.3)

        packet = struct.pack(PACKET_FORMAT, seq % 65536, t, value)

        should_drop = loss_enabled.is_set() and np.random.random() < 0.10

        if not should_drop:
            try:
                sock.sendto(packet, (HOST, PORT))
            except Exception:
                pass

        with stats_lock:
            stats["sent"] += 1
            if should_drop:
                stats["dropped"] += 1

        seq += 1
        time.sleep(interval)

    sock.close()


def receive_packets(sock):
    while running:
        try:
            sock.settimeout(0.1)
            data, _ = sock.recvfrom(1024)

            if len(data) < PACKET_SIZE:
                continue

            seq, t, value = struct.unpack(PACKET_FORMAT, data[:PACKET_SIZE])

            with stats_lock:
                if stats["last_seq"] >= 0:
                    expected = (stats["last_seq"] + 1) % 65536

                    if seq != expected:
                        gap = (seq - stats["last_seq"] - 1) % 65536

                        if 0 < gap < 1000:
                            stats["dropped"] += gap

                stats["last_seq"] = seq
                stats["received"] += 1

                if stats["sent"] > 0:
                    stats["loss_pct"] = 100.0 * stats["dropped"] / max(stats["sent"], 1)

            with recv_lock:
                recv_buffer.append((t, value))

        except socket.timeout:
            continue
        except Exception:
            continue


def run_oscilloscope():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.bind((HOST, PORT))
    except OSError as e:
        print(f"ERROR: Cannot bind to {HOST}:{PORT} — {e}")
        print("Is another process using port 5005? Try a different PORT in the script.")
        sys.exit(1)

    recv_thread = threading.Thread(target=receive_packets, args=(sock,), daemon=True)
    recv_thread.start()

    fig, (ax_wave, ax_stats) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    fig.suptitle("UDPOscilloscope — Day 21", fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#0a0a0f")

    ax_wave.set_xlim(0, BUFFER_SIZE)
    ax_wave.set_ylim(-1.8, 1.8)
    ax_wave.set_facecolor("#0d0d14")
    ax_wave.set_ylabel("Signal Value", color="white")
    ax_wave.set_title("Live Sensor Stream (UDP)", color="white")
    ax_wave.grid(True, alpha=0.2, color="#333")
    ax_wave.tick_params(colors="white")

    for spine in ax_wave.spines.values():
        spine.set_edgecolor("#333")

    wave_line, = ax_wave.plot([], [], color="#22c55e", linewidth=1.5)
    ax_wave.axhline(0, color="#333", linewidth=0.8, linestyle="--")

    status_text = ax_wave.text(
        0.02,
        0.92,
        "",
        transform=ax_wave.transAxes,
        fontsize=10,
        color="white",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.9),
    )

    loss_text = ax_wave.text(
        0.75,
        0.92,
        "",
        transform=ax_wave.transAxes,
        fontsize=10,
        color="#ef4444",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.9),
    )

    ax_stats.set_facecolor("#0d0d14")
    ax_stats.set_xlim(0, BUFFER_SIZE)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")

    stats_text_obj = ax_stats.text(
        0.5,
        0.5,
        "",
        transform=ax_stats.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#aaa",
        fontfamily="monospace",
    )

    plt.tight_layout()

    loss_history = collections.deque([0.0] * 50, maxlen=50)

    def update(frame):
        with recv_lock:
            buf = list(recv_buffer)

        if not buf:
            wave_line.set_data([], [])
            return wave_line, status_text, loss_text, stats_text_obj

        values = [v for _, v in buf]
        x = np.arange(len(values))
        wave_line.set_data(x, values)

        with stats_lock:
            sent = stats["sent"]
            received = stats["received"]
            dropped = stats["dropped"]
            loss = stats["loss_pct"]

        loss_history.append(loss)

        if loss > 20:
            wave_line.set_color("#ef4444")
        elif loss > 5:
            wave_line.set_color("#f59e0b")
        else:
            wave_line.set_color("#22c55e")

        mode_parts = []

        if loss_enabled.is_set():
            mode_parts.append("LOSS:ON")

        if noise_enabled.is_set():
            mode_parts.append("NOISE:ON")

        mode_str = " | ".join(mode_parts) if mode_parts else "CLEAN"

        status_text.set_text(f"Rate: {SEND_RATE_HZ} Hz | Mode: {mode_str}")

        if loss > 0.1:
            loss_text.set_text(f"⚠ LOSS: {loss:.1f}%")
            loss_text.set_color("#ef4444")
        else:
            loss_text.set_text("● CLEAN")
            loss_text.set_color("#22c55e")

        sparkline = "".join(
            "█" if l > 10 else ("▄" if l > 3 else "▁")
            for l in list(loss_history)[-30:]
        )

        stats_text_obj.set_text(
            f"Sent: {sent:,}  Received: {received:,}  "
            f"Dropped: {dropped:,}  Loss: {loss:.1f}%\n"
            f"Loss history: {sparkline}"
        )

        return wave_line, status_text, loss_text, stats_text_obj

    def on_key(event):
        global running

        if event.key == "l":
            if loss_enabled.is_set():
                loss_enabled.clear()
                print("📶 Packet loss: OFF")
            else:
                loss_enabled.set()
                print("📶 Packet loss: ON (10% drop rate)")

        elif event.key == "n":
            if noise_enabled.is_set():
                noise_enabled.clear()
                print("〜 Noise: OFF")
            else:
                noise_enabled.set()
                print("〜 Noise: ON")

        elif event.key == "q":
            running = False
            plt.close()

    fig.canvas.mpl_connect("key_press_event", on_key)

    animation.FuncAnimation(
        fig,
        update,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    print("\n" + "=" * 55)
    print("  📡 UDPOscilloscope — Day 21")
    print("=" * 55)
    print(f"  Sender:   {HOST}:{PORT} @ {SEND_RATE_HZ} Hz")
    print(f"  Buffer:   {BUFFER_SIZE} samples")
    print(f"  Packet:   {PACKET_SIZE} bytes ({PACKET_FORMAT})")
    print()
    print("  Controls (click plot first):")
    print("  'l' → toggle 10% packet loss")
    print("  'n' → toggle noise injection")
    print("  'q' → quit")
    print()
    print("  Experiment: press 'l' and watch the waveform degrade.")
    print("  This is what a noisy UART line looks like.\n")

    plt.show()
    sock.close()


if __name__ == "__main__":
    send_thread = threading.Thread(target=sender_thread, daemon=True)
    send_thread.start()

    time.sleep(0.2)

    try:
        run_oscilloscope()
    except KeyboardInterrupt:
        pass
    finally:
        running = False

    print("\nUDPOscilloscope ended.")
    print("Week 3 complete! See you Monday for Week 4 🎉")
