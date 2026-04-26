import time
import threading
import collections
import csv
import sys
from datetime import datetime

import numpy as np
import psutil
import cv2
import pyaudio

try:
    from rich.live import Live
    from rich.table import Table
    from rich.console import Console
    from rich.panel import Panel
    from rich import box

    console = Console()
except ImportError:
    print("pip install rich")
    sys.exit(1)


SAMPLE_RATE_HZ = 10
RING_BUFFER_SIZE = 200
DISPLAY_HISTORY = 20


class SensorChannel:
    """One channel of the DAQ: value, timestamp, history."""

    def __init__(self, name, unit, color="white"):
        self.name = name
        self.unit = unit
        self.color = color
        self.values = collections.deque(maxlen=RING_BUFFER_SIZE)
        self.timestamps = collections.deque(maxlen=RING_BUFFER_SIZE)
        self.lock = threading.Lock()

    def push(self, value):
        with self.lock:
            self.values.append(float(value))
            self.timestamps.append(time.time())

    def latest(self):
        with self.lock:
            return self.values[-1] if self.values else 0.0

    def history(self, n=DISPLAY_HISTORY):
        with self.lock:
            return list(self.values)[-n:]

    def mean(self):
        with self.lock:
            vals = list(self.values)
        return float(np.mean(vals)) if vals else 0.0

    def std(self):
        with self.lock:
            vals = list(self.values)
        return float(np.std(vals)) if len(vals) > 1 else 0.0


channels = {
    "mic_rms": SensorChannel("Mic RMS", "amplitude", "cyan"),
    "cam_motion": SensorChannel("Cam Motion", "px/frame", "green"),
    "cpu_pct": SensorChannel("CPU %", "%", "yellow"),
    "net_kbps": SensorChannel("Network", "KB/s", "magenta"),
}

running = True


def mic_reader():
    try:
        pa = pyaudio.PyAudio()
        dev = None

        for i in range(pa.get_device_count()):
            if pa.get_device_info_by_index(i)["maxInputChannels"] > 0:
                dev = i
                break

        if dev is None:
            return

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=dev,
            frames_per_buffer=1600,
        )

        while running:
            try:
                data = stream.read(1600, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)
                rms = float(np.sqrt(np.mean(samples ** 2)))
                channels["mic_rms"].push(rms)
            except Exception:
                channels["mic_rms"].push(0.0)

            time.sleep(1.0 / SAMPLE_RATE_HZ)

        stream.stop_stream()
        stream.close()
        pa.terminate()

    except Exception:
        while running:
            channels["mic_rms"].push(0.0)
            time.sleep(1.0 / SAMPLE_RATE_HZ)


def cam_reader():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        while running:
            channels["cam_motion"].push(0.0)
            time.sleep(1.0 / SAMPLE_RATE_HZ)
        return

    prev_gray = None

    while running:
        ret, frame = cap.read()

        if not ret:
            channels["cam_motion"].push(0.0)
            time.sleep(1.0 / SAMPLE_RATE_HZ)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            channels["cam_motion"].push(magnitude)
        else:
            channels["cam_motion"].push(0.0)

        prev_gray = gray
        time.sleep(1.0 / SAMPLE_RATE_HZ)

    cap.release()


def system_reader():
    while running:
        channels["cpu_pct"].push(psutil.cpu_percent(interval=None))
        time.sleep(1.0 / SAMPLE_RATE_HZ)


def network_reader():
    previous = psutil.net_io_counters()
    previous_total = previous.bytes_sent + previous.bytes_recv
    previous_time = time.time()

    while running:
        time.sleep(1.0 / SAMPLE_RATE_HZ)

        current = psutil.net_io_counters()
        current_total = current.bytes_sent + current.bytes_recv
        current_time = time.time()

        elapsed = max(current_time - previous_time, 1e-6)
        kbps = ((current_total - previous_total) / 1024.0) / elapsed

        channels["net_kbps"].push(kbps)

        previous_total = current_total
        previous_time = current_time


ANOMALY_SIGMA = 3.0
anomaly_log = collections.deque(maxlen=5)


def check_anomalies():
    alerts = []

    for name, ch in channels.items():
        if len(ch.values) < 20:
            continue

        val = ch.latest()
        mean = ch.mean()
        std = ch.std()

        if std > 1e-6 and abs(val - mean) > ANOMALY_SIGMA * std:
            direction = "↑" if val > mean else "↓"
            alerts.append(f"{ch.name} {direction} {val:.3f}")

    return alerts


def sparkline(values, width=20):
    chars = " ▁▂▃▄▅▆▇█"

    if not values or max(values) == min(values):
        return "─" * width

    lo, hi = min(values), max(values)
    result = ""

    for v in values[-width:]:
        idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
        result += chars[idx]

    return result.ljust(width)


def build_dashboard():
    table = Table(
        title="SensorLogger — DAQ Dashboard",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Channel", style="bold", width=14)
    table.add_column("Latest", justify="right", width=14)
    table.add_column("Mean", justify="right", width=10)
    table.add_column("Std", justify="right", width=8)
    table.add_column("Trend", width=22)
    table.add_column("Status", width=10)

    for name, ch in channels.items():
        val = ch.latest()
        mean = ch.mean()
        std = ch.std()
        hist = ch.history()
        spark = sparkline(hist)

        is_anomaly = (
            std > 1e-6
            and len(ch.values) >= 20
            and abs(val - mean) > ANOMALY_SIGMA * std
        )

        status = "[red]⚠ SPIKE[/red]" if is_anomaly else "[green]OK[/green]"

        table.add_row(
            f"[{ch.color}]{ch.name}[/{ch.color}]",
            f"[{ch.color}]{val:.4f}[/{ch.color}] {ch.unit}",
            f"{mean:.4f}",
            f"{std:.4f}",
            f"[{ch.color}]{spark}[/{ch.color}]",
            status,
        )

    alerts = check_anomalies()

    if alerts:
        anomaly_log.extend(alerts)

    sample_count = min(len(ch.values) for ch in channels.values())

    stats_text = (
        f"[dim]Samples: {sample_count} | "
        f"Buffer: {RING_BUFFER_SIZE} | "
        f"Rate: {SAMPLE_RATE_HZ} Hz | "
        f"Press 's' to export CSV, 'q' to quit[/dim]"
    )

    if anomaly_log:
        alert_text = "[red]⚠ Recent anomalies: " + " | ".join(list(anomaly_log)[-3:]) + "[/red]"
    else:
        alert_text = "[green]No anomalies detected[/green]"

    return Panel(
        table,
        subtitle=f"{stats_text}\n{alert_text}",
        border_style="cyan",
    )


def export_csv():
    fname = f"sensorlog_{datetime.now().strftime('%H%M%S')}.csv"

    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["timestamp"] + list(channels.keys())
        writer.writerow(header)

        min_len = min(len(ch.timestamps) for ch in channels.values())
        first_channel = list(channels.values())[0]

        for i in range(min_len):
            row = [list(first_channel.timestamps)[i]]

            for ch in channels.values():
                row.append(list(ch.values)[i])

            writer.writerow(row)

    console.print(f"\n[green]✓ Exported {fname} ({min_len} rows)[/green]")
    return fname


def main():
    global running

    console.print("\n[bold cyan]📊 SensorLogger — Day 20[/bold cyan]")
    console.print("[dim]Starting sensor threads...[/dim]\n")

    threads = [
        threading.Thread(target=mic_reader, daemon=True),
        threading.Thread(target=cam_reader, daemon=True),
        threading.Thread(target=system_reader, daemon=True),
        threading.Thread(target=network_reader, daemon=True),
    ]

    for thread in threads:
        thread.start()

    time.sleep(1.5)

    console.print("[green]✓ All sensors active[/green]")
    console.print("[dim]Press 's' + Enter to export CSV. Press 'q' + Enter to quit.[/dim]\n")

    export_flag = threading.Event()

    def input_listener():
        global running

        while running:
            try:
                cmd = input().strip().lower()

                if cmd == "s":
                    export_flag.set()
                elif cmd == "q":
                    running = False
            except Exception:
                break

    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()

    try:
        with Live(build_dashboard(), refresh_per_second=4, console=console) as live:
            while running:
                live.update(build_dashboard())

                if export_flag.is_set():
                    export_csv()
                    export_flag.clear()

                time.sleep(0.25)

    except KeyboardInterrupt:
        pass
    finally:
        running = False

    console.print("\n[bold]SensorLogger ended.[/bold]")
    console.print("See you tomorrow for Day 21!")


if __name__ == "__main__":
    main()
