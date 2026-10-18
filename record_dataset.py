#!/usr/bin/env python3
"""
SignSpeak Dataset Collection Script
Receives CSV rows from ESP32 over USB Serial, validates, and writes to dataset.csv.
User types commands (letters) in Python terminal, not Serial Monitor.
"""

import serial
import serial.tools.list_ports
import csv
import sys
import threading
from pathlib import Path
from datetime import datetime

# ============= CONFIG =============
DATASET_CSV = "dataset.csv"
BAUD_RATE = 115200

# Expected classes (26 letters + control gestures)
LETTERS = [chr(ord('A') + i) for i in range(26)]
CONTROL_GESTURES = ['NEUTRAL', 'SPEAK']
VALID_LABELS = LETTERS + CONTROL_GESTURES

TARGET_REPS_PER_LABEL = 40
SAMPLES_PER_REP = 50

# CSV columns: 3 meta + the right glove's channels (13 with firmware 2.0:
# 5 flex, 3 accel, 3 gyro, 2 touch). One place to change: features.py.
from features import RIGHT_CHANNELS
CSV_COLUMNS = ['label', 'rep', 'sample'] + RIGHT_CHANNELS

# Any of these means the firmware finished calibrating and is idling in
# loop(). Keep this list permissive: firmware banner text changes, and a
# missed match leaves the script hanging with no way to type a label.
READY_MARKERS = (
    "READY FOR DATA COLLECTION",
    "READY",
    "Calibration complete",
)

# ============= HELPER FUNCTIONS =============

def find_esp32_port():
    """Auto-detect ESP32 serial port."""
    ports = serial.tools.list_ports.comports()
    esp32_ports = [
        p.device for p in ports
        if 'CH340' in p.description or 'USB' in p.description
        or 'FTDI' in p.description or 'CP210' in p.description
        or 'Serial' in p.description
    ]

    if not esp32_ports:
        # Fall back to every port rather than giving up.
        esp32_ports = [p.device for p in ports]

    if not esp32_ports:
        print("No serial ports found. Check the USB connection.")
        return None

    if len(esp32_ports) == 1:
        print(f"Found ESP32 on {esp32_ports[0]}")
        return esp32_ports[0]

    print("Multiple serial ports found:")
    for i, p in enumerate(esp32_ports):
        print(f"  {i+1}. {p}")
    choice = input("Select port (number): ").strip()
    try:
        return esp32_ports[int(choice) - 1]
    except Exception:
        print("Invalid choice.")
        return None

def get_existing_data():
    """Load existing dataset to resume from."""
    if not Path(DATASET_CSV).exists():
        return {}

    existing = {}
    with open(DATASET_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label']
            rep = int(row['rep'])
            if label not in existing:
                existing[label] = set()
            existing[label].add(rep)
    return existing

def init_dataset_csv():
    """Create or verify dataset CSV header."""
    if not Path(DATASET_CSV).exists():
        with open(DATASET_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        print(f"Created {DATASET_CSV}")

def validate_row(parts):
    """Validate a CSV row from ESP32."""
    if len(parts) != len(CSV_COLUMNS):
        return False, (f"Column count mismatch (got {len(parts)}, expected "
                       f"{len(CSV_COLUMNS)}) - is firmware 2.0 flashed and does "
                       f"its touch column count match features.py?")

    label = parts[0]
    if label not in VALID_LABELS:
        return False, f"Invalid label: {label}"

    try:
        rep = int(parts[1])
        sample = int(parts[2])
        if rep < 1 or sample < 0 or sample >= SAMPLES_PER_REP:
            return False, "Rep or sample out of range"

        # Validate numeric values (flex, accel, gyro)
        for i in range(3, len(parts)):
            float(parts[i])
    except ValueError as e:
        return False, f"Parse error: {str(e)}"

    return True, None

def print_progress(existing_data):
    """Print collection progress."""
    total_target = len(VALID_LABELS) * TARGET_REPS_PER_LABEL
    total_collected = sum(len(reps) for reps in existing_data.values())

    print("\n" + "="*70)
    print(f"Collection Progress: {total_collected}/{total_target} reps")
    print("="*70)

    for label in sorted(VALID_LABELS):
        reps_collected = len(existing_data.get(label, set()))
        pct = (reps_collected / TARGET_REPS_PER_LABEL) * 100
        bar_len = 22
        filled = int(bar_len * reps_collected / TARGET_REPS_PER_LABEL)
        bar = "#" * filled + "-" * (bar_len - filled)

        print(f"{label:10s} [{bar}] {reps_collected:2d}/{TARGET_REPS_PER_LABEL} ({pct:5.1f}%)")

    print("="*70 + "\n")

def main():
    print("\n" + "="*60)
    print("SignSpeak Dataset Collection")
    print("="*60)

    # Initialize dataset CSV
    init_dataset_csv()

    # Load existing data
    existing_data = get_existing_data()
    print_progress(existing_data)

    # Find and open serial port
    port = find_esp32_port()
    if not port:
        return

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        print(f"Connected to {port} at {BAUD_RATE} baud")
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        return

    # ===== CALIBRATION PHASE =====
    # The ESP32 resets when the port opens, so it runs setup() now.
    # Print everything it says until it signals that it is ready.
    print("\n" + "="*60)
    print("CALIBRATION - follow the prompts below")
    print("="*60)

    calib_start = datetime.now()
    last_line_at = datetime.now()
    ready = False

    while not ready:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if line:
                last_line_at = datetime.now()

                # Firmware is waiting on us - block until the user hits Enter
                if "PRESS_ENTER" in line:
                    prompt = line.split("PRESS_ENTER", 1)[1].lstrip(": ").strip()
                    input(f"\n>>> {prompt} ")
                    ser.write(b"\n")
                    ser.flush()
                    calib_start = datetime.now()  # reset the timeout
                    last_line_at = datetime.now()
                    continue

                print(f"  {line}", flush=True)

                if any(m in line for m in READY_MARKERS):
                    ready = True
                    # Drain the rest of the banner so it does not get
                    # mistaken for data later.
                    ser.timeout = 0.3
                    while True:
                        extra = ser.readline().decode('utf-8', errors='ignore').strip()
                        if not extra:
                            break
                        print(f"  {extra}", flush=True)
                    ser.timeout = 2
                    break
                continue

            # No line came back. If the board has been silent for a while
            # after printing something, assume it finished calibrating.
            quiet_s = (datetime.now() - last_line_at).total_seconds()
            if quiet_s > 5:
                print("\n  (board is quiet - assuming calibration is done)")
                ready = True
                break

        except Exception:
            pass

        # Safety timeout so we never hang forever
        if (datetime.now() - calib_start).total_seconds() > 180:
            print("\nTimed out waiting for calibration to finish.")
            print("  Continuing anyway - try typing a label below.")
            break

    print("\n" + "="*60)
    print("Ready to collect. Type gestures in THIS TERMINAL as:")
    print("  A          (letter A, continues from the last saved rep)")
    print("  A,5        (force rep 5 for A, if needed)")
    print("\nCommands (spelled out, so they do not clash with letters):")
    print("  QUIT       (stop and save)")
    print("  PROGRESS   (show collection status)")
    print("\n  Note: typing Q or P records the LETTER Q or P.")
    print("="*60 + "\n")

    # State
    sample_count = 0
    current_label = None
    current_rep = None
    buffer = []
    quit_flag = False

    def next_rep_for(label):
        """
        Next unused rep number for this label.

        The ESP32 keeps its own rep counters in RAM and they reset to zero
        every time it reboots, which happens whenever the serial port is
        opened. So the firmware always thinks it is starting fresh. Python
        holds the real history, so it decides the rep number and sends the
        explicit LABEL,REP form instead of a bare label.
        """
        used = existing_data.get(label, set())
        return (max(used) + 1) if used else 1

    # Read input in a separate thread
    def read_input():
        nonlocal quit_flag
        while not quit_flag:
            try:
                user_input = input().strip().upper()
                if not user_input:
                    continue

                # Control commands must NOT collide with letter labels.
                # 'Q' and 'P' are letters in the alphabet, so the commands
                # are spelled out instead.
                if user_input in ('QUIT', 'EXIT', ':Q'):
                    quit_flag = True
                    break
                if user_input in ('PROGRESS', 'STATUS', ':P'):
                    print_progress(existing_data)
                    continue

                if ',' in user_input:
                    # Manual override, pass straight through.
                    label = user_input.split(',', 1)[0].strip()
                    try:
                        rep = int(user_input.split(',', 1)[1])
                    except ValueError:
                        print(f"  bad rep number in '{user_input}'")
                        continue
                    if rep in existing_data.get(label, set()):
                        print(f"  WARNING: {label} rep {rep} already exists "
                              f"and will be duplicated")
                    command = f"{label},{rep}"
                else:
                    label = user_input
                    if label not in VALID_LABELS:
                        print(f"  '{label}' is not a valid label")
                        continue
                    rep = next_rep_for(label)
                    command = f"{label},{rep}"

                ser.write(f"{command}\n".encode())
                ser.flush()
                print(f"-> {label} rep {rep}", flush=True)
            except EOFError:
                quit_flag = True
            except Exception as e:
                print(f"Input error: {e}")

    input_thread = threading.Thread(target=read_input, daemon=True)
    input_thread.start()

    # Main loop: read from ESP32
    while not quit_flag:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:
                continue

            # Check for "END" marker
            if line == "END":
                if buffer:
                    # Write buffered samples to CSV
                    with open(DATASET_CSV, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                        for row_dict in buffer:
                            writer.writerow(row_dict)

                    print(f"Logged {current_label} rep {current_rep} ({len(buffer)} samples)")
                    sample_count += len(buffer)
                    buffer = []

                    # Update existing data
                    if current_label not in existing_data:
                        existing_data[current_label] = set()
                    existing_data[current_label].add(current_rep)

                    print_progress(existing_data)
                continue

            # Parse CSV row from ESP32
            parts = line.split(',')
            valid, error_msg = validate_row(parts)

            if not valid:
                # Not a data row. Anything without commas is a firmware
                # status/prompt message - show it as-is. Malformed data
                # rows (right shape, bad values) get flagged.
                if len(parts) < 5:
                    print(f"  {line}", flush=True)
                else:
                    print(f"Invalid row: {error_msg}")
                continue

            # Extract data
            label = parts[0]
            rep = int(parts[1])
            sample = int(parts[2])

            # Track current gesture
            if label != current_label or rep != current_rep:
                current_label = label
                current_rep = rep

            # Build row dict
            row_dict = {}
            for i, col in enumerate(CSV_COLUMNS):
                if i < 3:
                    row_dict[col] = parts[i]
                else:
                    row_dict[col] = float(parts[i])

            buffer.append(row_dict)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            quit_flag = True
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    ser.close()
    print(f"\nDataset closed. Total samples logged: {sample_count}")
    print(f"Saved to {DATASET_CSV}")

if __name__ == "__main__":
    main()