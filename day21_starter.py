import socket
import struct
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections
import sys

# ============================================================
# NETWORK CONFIG
# ============================================================

HOST = "127.0.0.1"
PORT = 5005
SEND_RATE_HZ = 200
BUFFER_SIZE = 512

# seq (uint16), time (float), value (double)
PACKET_FORMAT = "!Hfd"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# ============================================================
# SHARED STATE
# ============================================================

recv_buffer = collections.deque(maxlen=BUFFER_SIZE)
recv_lock = threading.Lock()

stats = {
    "sent": 0,
    "received": 0,
    "dropped": 0,
    "loss_pct": 0.0,
    "last_seq": None,
}
stats_lock = threading.Lock()

loss_enabled = threading.Event()
noise_enabled = threading.Event()
running = True

# ============================================================
# SENDER
# ============================================================

def sender_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    seq = 0
    start_time = time.time()
    interval = 1.0 / SEND_RATE_HZ

    while running:
        t = time.time() - start_time

        value = np.sin(2 * np.pi * 2 * t)

        if noise_enabled.is_set():
            value += np.random.normal(0, 0.3)

        packet = struct.pack(PACKET_FORMAT, seq % 65536, t, value)

        # Packet loss simulator (10%)
        drop = loss_enabled.is_set() and np.random.random() < 0.10

        if not drop:
            try:
                sock.sendto(packet, (HOST, PORT))
            except:
                pass

        with stats_lock:
            stats["sent"] += 1
            if drop:
                stats["dropped"] += 1

        seq += 1
        time.sleep(interval)

    sock.close()

# ============================================================
# RECEIVER
# ============================================================

def receive_packets(sock):
    while running:
        try:
            sock.settimeout(0.1)
            data, _ = sock.recvfrom(1024)

            if len(data) < PACKET_SIZE:
                continue

            seq, t, value = struct.unpack(PACKET_FORMAT, data[:PACKET_SIZE])

            with stats_lock:
                if stats["last_seq"] is not None:
                    expected = (stats["last_seq"] + 1) % 65536

                    if seq != expected:
                        gap = (seq - stats["last_seq"] - 1) % 65536

                        if 0 < gap < 1000:
                            stats["dropped"] += gap

                stats["last_seq"] = seq
                stats["received"] += 1

                # Packet loss percentage
                if stats["sent"] > 0:
                    stats["loss_pct"] = (
                        stats["dropped"] / stats["sent"]
                    ) * 100.0

            with recv_lock:
                recv_buffer.append((t, value))

        except socket.timeout:
            continue
        except:
            continue

# ============================================================
# VISUALIZATION
# ============================================================

def run_oscilloscope():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.bind((HOST, PORT))
    except OSError as e:
        print("Port error:", e)
        sys.exit(1)

    threading.Thread(target=receive_packets, args=(sock,), daemon=True).start()

    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    ax.set_ylim(-2, 2)
    ax.set_xlim(0, BUFFER_SIZE)

    def update(frame):
        with recv_lock:
            data = list(recv_buffer)

        if not data:
            return line,

        values = [v for _, v in data]
        x = np.arange(len(values))

        line.set_data(x, values)

        with stats_lock:
            loss = stats["loss_pct"]

        ax.set_title(f"Packet Loss: {loss:.2f}%")

        return line,

    def on_key(event):
        global running

        if event.key == 'l':
            if loss_enabled.is_set():
                loss_enabled.clear()
                print("Loss OFF")
            else:
                loss_enabled.set()
                print("Loss ON (10%)")

        elif event.key == 'n':
            if noise_enabled.is_set():
                noise_enabled.clear()
                print("Noise OFF")
            else:
                noise_enabled.set()
                print("Noise ON")

        elif event.key == 'q':
            running = False
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)

    animation.FuncAnimation(fig, update, interval=50)
    plt.show()

    sock.close()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    threading.Thread(target=sender_thread, daemon=True).start()
    time.sleep(0.2)

    try:
        run_oscilloscope()
    except KeyboardInterrupt:
        pass
    finally:
        running = False

    print("Finished.")
