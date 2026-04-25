import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SCL_FREQ = 100_000
SAMPLES_PER_BIT = 20


def bits_to_waveform(bits):
    sda = []
    scl = []

    for bit in bits:
        sda += [bit] * SAMPLES_PER_BIT
        scl += (
            [0] * (SAMPLES_PER_BIT // 4)
            + [1] * (SAMPLES_PER_BIT // 2)
            + [0] * (SAMPLES_PER_BIT // 4)
        )

    return sda, scl


def generate_ack(ack=True):
    sda_val = 0 if ack else 1
    sda = [sda_val] * SAMPLES_PER_BIT
    scl = (
        [0] * (SAMPLES_PER_BIT // 4)
        + [1] * (SAMPLES_PER_BIT // 2)
        + [0] * (SAMPLES_PER_BIT // 4)
    )
    return sda, scl


def encode_i2c_transaction(device_addr, register_addr, data_bytes, read=False):
    segments = []

    segments.append(("START", [1, 1, 0, 0], [1, 1, 1, 0], "START"))

    addr_bits = [(device_addr >> (6 - i)) & 1 for i in range(7)]
    rw_bit = 1 if read else 0
    addr_byte = addr_bits + [rw_bit]

    sda_seq, scl_seq = bits_to_waveform(addr_byte)
    segments.append(
        (
            "ADDR+RW",
            sda_seq,
            scl_seq,
            f"ADDR=0x{device_addr:02X} {'R' if read else 'W'}",
        )
    )

    ack_sda, ack_scl = generate_ack(ack=True)
    segments.append(("ACK", ack_sda, ack_scl, "ACK"))

    reg_bits = [(register_addr >> (7 - i)) & 1 for i in range(8)]
    sda_seq, scl_seq = bits_to_waveform(reg_bits)
    segments.append(("REG", sda_seq, scl_seq, f"REG=0x{register_addr:02X}"))

    ack_sda, ack_scl = generate_ack(ack=True)
    segments.append(("ACK", ack_sda, ack_scl, "ACK"))

    for i, byte in enumerate(data_bytes):
        data_bits = [(byte >> (7 - j)) & 1 for j in range(8)]
        sda_seq, scl_seq = bits_to_waveform(data_bits)

        segments.append((f"DATA[{i}]", sda_seq, scl_seq, f"DATA=0x{byte:02X}"))

        is_last = i == len(data_bytes) - 1
        ack = not (read and is_last)

        ack_sda, ack_scl = generate_ack(ack=ack)
        label = "ACK" if ack else "NACK"
        segments.append((label, ack_sda, ack_scl, label))

    segments.append(("STOP", [0, 0, 1, 1], [0, 1, 1, 1], "STOP"))

    return segments


def decode_i2c_segments(segments):
    decoded = []

    for label, sda_bits, scl_bits, annotation in segments:
        if label in ("START", "STOP"):
            decoded.append(f"[{label}]")

        elif label in ("ACK", "NACK", "STRETCH+ACK"):
            decoded.append(f"  → {annotation}")

        else:
            bits = []
            i = 0

            while i < len(sda_bits):
                if i < len(scl_bits) and scl_bits[i] == 1:
                    bits.append(sda_bits[i])

                    while i < len(scl_bits) and scl_bits[i] == 1:
                        i += 1
                else:
                    i += 1

            if len(bits) >= 8:
                byte_val = 0
                for bit in bits[:8]:
                    byte_val = (byte_val << 1) | bit

                decoded.append(
                    f"  {label}: 0x{byte_val:02X} = {byte_val:08b}b = {byte_val}d"
                )
            else:
                decoded.append(f"  {label}: {bits}")

    return decoded


def simulate_nack():
    print("\n--- TODO #1: NACK simulation ---")

    segments = encode_i2c_transaction(0x48, 0x1A, [0x42], read=False)

    nack_sda, nack_scl = generate_ack(ack=False)
    segments[2] = ("NACK", nack_sda, nack_scl, "NACK: device not found")

    print_transaction(segments)
    print("I2C ERROR: Device not found at address 0x48")

    return segments


def apply_clock_stretching(segments, stretch_samples=60):
    print("\n--- TODO #2: Clock stretching simulation ---")

    stretched = []
    stretch_done = False

    for label, sda, scl, annotation in segments:
        if label == "ACK" and not stretch_done:
            sda = [sda[0]] * stretch_samples + sda
            scl = [0] * stretch_samples + scl
            label = "STRETCH+ACK"
            annotation = "STRETCH: slave holds SCL LOW, then ACK"
            stretch_done = True

        stretched.append((label, sda, scl, annotation))

    print_transaction(stretched)
    return stretched


def run_multibyte_transaction():
    print("\n--- TODO #3: Multi-byte MPU6050 transaction ---")

    device_addr = 0x68
    register_addr = 0x3B

    data = [
        0x03, 0xE8,
        0xFF, 0x01,
        0x00, 0x80,
        0x1A, 0x2B,
        0x00, 0x10,
        0xFF, 0xF0,
        0x01, 0x23,
    ]

    segments = encode_i2c_transaction(
        device_addr,
        register_addr,
        data,
        read=False,
    )

    print_transaction(segments)
    return segments


def build_full_waveform(segments):
    all_sda = []
    all_scl = []
    boundaries = [0]
    labels = []

    for label, sda, scl, annotation in segments:
        all_sda.extend(sda)
        all_scl.extend(scl)
        boundaries.append(len(all_sda))
        labels.append((boundaries[-2], annotation))

    return np.array(all_sda), np.array(all_scl), labels


def animate_transaction(segments, title="I2C Transaction"):
    sda_full, scl_full, labels = build_full_waveform(segments)
    n = len(sda_full)
    window = 200

    fig, (ax_scl, ax_sda) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax_scl.set_ylim(-0.3, 1.5)
    ax_scl.set_ylabel("SCL", fontweight="bold")
    ax_scl.set_yticks([0, 1])
    ax_scl.set_yticklabels(["LOW", "HIGH"])
    ax_scl.grid(True, alpha=0.2)
    ax_scl.set_facecolor("#0d0d14")

    scl_line, = ax_scl.step([], [], color="#f59e0b", linewidth=2, where="post")

    ax_sda.set_ylim(-0.3, 1.5)
    ax_sda.set_ylabel("SDA", fontweight="bold")
    ax_sda.set_xlabel("Sample index")
    ax_sda.set_yticks([0, 1])
    ax_sda.set_yticklabels(["LOW", "HIGH"])
    ax_sda.grid(True, alpha=0.2)
    ax_sda.set_facecolor("#0d0d14")

    sda_line, = ax_sda.step([], [], color="#22c55e", linewidth=2, where="post")

    ann_text = ax_scl.text(
        0.02,
        1.2,
        "",
        transform=ax_scl.transAxes,
        fontsize=10,
        color="white",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#333", alpha=0.8),
    )

    fig.patch.set_facecolor("#0a0a0f")

    for ax in (ax_scl, ax_sda):
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")

        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    def update(frame):
        start = min(frame * 4, max(0, n - window))
        end = start + window
        x = np.arange(start, end)

        scl_slice = scl_full[start:end]
        sda_slice = sda_full[start:end]

        if len(scl_slice) < window:
            scl_slice = np.pad(scl_slice, (0, window - len(scl_slice)))
            sda_slice = np.pad(sda_slice, (0, window - len(sda_slice)))

        scl_line.set_data(x, scl_slice)
        sda_line.set_data(x, sda_slice)

        ax_scl.set_xlim(start, end)
        ax_sda.set_xlim(start, end)

        current_label = ""
        for pos, lbl in labels:
            if pos <= start + window // 2:
                current_label = lbl

        ann_text.set_text(current_label)

        return scl_line, sda_line, ann_text

    frames = max(1, (n - window) // 4 + 20)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=60,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()

    return ani


def print_transaction(segments):
    print("\n" + "=" * 60)
    print("  I2C Transaction text mode")
    print("=" * 60)

    for label, sda_bits, scl_bits, annotation in segments:
        sda_display = "".join(str(b) for b in sda_bits[:12])
        scl_display = "".join(str(b) for b in scl_bits[:12])

        print(
            f"  {label:<12} | SCL: {scl_display}... "
            f"| SDA: {sda_display}... | {annotation}"
        )

    print("\nDecoded:")

    decoded = decode_i2c_segments(segments)
    for line in decoded:
        print(f"  {line}")

    print()


def main():
    print("\n" + "=" * 60)
    print("  I2CPlayground — Day 19")
    print("=" * 60)
    print()

    device_addr = 0x48
    register = 0x1A
    data = [0x42]

    print("Encoding normal write transaction:")
    print(
        f"Device: 0x{device_addr:02X} | "
        f"Register: 0x{register:02X} | "
        f"Data: {[hex(b) for b in data]}"
    )

    segments = encode_i2c_transaction(device_addr, register, data, read=False)

    print_transaction(segments)

    print("Key I2C facts:")
    print("  - ACK = SDA pulled LOW by slave during 9th bit")
    print("  - NACK = SDA stays HIGH")
    print("  - START: SDA goes LOW while SCL is HIGH")
    print("  - STOP: SDA goes HIGH while SCL is HIGH")
    print("  - MSB first: bit 7 goes on the wire before bit 0")

    simulate_nack()

    stretched_segments = apply_clock_stretching(segments)

    mpu_segments = run_multibyte_transaction()

    print("\nLaunching animated waveform...")
    animate_transaction(
        segments,
        title="Normal I2C Write Transaction",
    )

    animate_transaction(
        stretched_segments,
        title="I2C Clock Stretching Demo",
    )

    animate_transaction(
        mpu_segments,
        title="MPU6050 14-byte I2C Burst",
    )

    print("\nSee you tomorrow for Day 20!")


if __name__ == "__main__":
    main()
