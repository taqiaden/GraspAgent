#!/usr/bin/env python3
"""
input_event_logger.py

Listen for keyboard and mouse events and append them (with ISO timestamps) to a text file.
Each run creates a new file with incrementing suffix, e.g., input_events_1.log, input_events_2.log.

Requirements:
    pip install pynput

Run:
    python3 input_event_logger.py
    python3 input_event_logger.py --no-print

Use responsibly and only where you have permission.
"""

import sys
import argparse
import threading
from pathlib import Path
from datetime import datetime

from pynput import keyboard, mouse

# ---------- CONFIG ----------
LOG_DIR = Path(".")
LOG_BASENAME = "input_events"   # base, suffix and .log appended automatically
PRINT_TO_STDOUT = True          # default; can be overridden with CLI --no-print
# -----------------------------

LOG_LOCK = threading.Lock()

def now_iso():
    return datetime.now().isoformat(sep=' ', timespec='seconds')

def next_logfile(path: Path, base: str):
    """
    Find next logfile path in `path` with base name `base`.
    Existing files: base_1.log, base_2.log, ...
    Returns a Path for base_{N+1}.log where N is the max existing index (0 if none).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.glob(f"{base}_*.log"))
    max_idx = 0
    for p in existing:
        name = p.stem  # e.g., "input_events_3"
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[0] == base:
            try:
                idx = int(parts[1])
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    new_idx = max_idx + 1
    return path / f"{base}_{new_idx}.log"

def open_logfile(path: Path):
    """Open file for append (creates if missing). Return Path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Touch the file once so it exists
    path.touch(exist_ok=True)
    return path

def write_log(line: str, logfile: Path):
    """Append a line to the log file (thread-safe)."""
    with LOG_LOCK:
        with logfile.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

def format_event_line(ts: str, event_type: str, details: str):
    return f"{ts}    {event_type:12s}  {details}"

# --- Keyboard callbacks ---
def make_key_callbacks(logfile: Path, print_out: bool):
    def on_key_press(key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)  # special keys
        line = format_event_line(now_iso(), "KEY_PRESS", f"{k}")
        if print_out:
            print(line)
        write_log(line, logfile)

    def on_key_release(key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        line = format_event_line(now_iso(), "KEY_RELEASE", f"{k}")
        if print_out:
            print(line)
        write_log(line, logfile)
        # Optionally stop on ESC:
        # if key == keyboard.Key.esc:
        #     return False

    return on_key_press, on_key_release

# --- Mouse callbacks ---
def make_mouse_callbacks(logfile: Path, print_out: bool):
    def on_move(x, y):
        line = format_event_line(now_iso(), "MOUSE_MOVE", f"x={x:.1f},y={y:.1f}")
        if print_out:
            print(line)
        write_log(line, logfile)

    def on_click(x, y, button, pressed):
        state = "PRESSED" if pressed else "RELEASED"
        line = format_event_line(now_iso(), "MOUSE_CLICK", f"button={button} {state} at x={x:.1f},y={y:.1f}")
        if print_out:
            print(line)
        write_log(line, logfile)

    def on_scroll(x, y, dx, dy):
        line = format_event_line(now_iso(), "MOUSE_SCROLL", f"dx={dx:.1f},dy={dy:.1f} at x={x:.1f},y={y:.1f}")
        if print_out:
            print(line)
        write_log(line, logfile)

    return on_move, on_click, on_scroll

def parse_args():
    p = argparse.ArgumentParser(description="Log keyboard and mouse events to a timestamped file.")
    p.add_argument("--no-print", action="store_false", help="Disable printing events to stdout.")
    p.add_argument("--log-dir", type=str, default=str(LOG_DIR), help="Directory where logs are saved.")
    p.add_argument("--base", type=str, default=LOG_BASENAME, help="Base name for the log files.")
    return p.parse_args()

def main():
    args = parse_args()
    print_out = not args.no_print

    logfile_path = next_logfile(Path(args.log_dir), args.base)
    logfile_path = open_logfile(logfile_path)
    if print_out:
        print(f"Logging input events to: {logfile_path.resolve()}")
    else:
        # still show the filename if you like; comment out the next line to be completely silent
        print(f"Logging events (printing disabled) to: {logfile_path.resolve()}")

    # Prepare callbacks bound to this logfile and print setting
    on_key_press, on_key_release = make_key_callbacks(logfile_path, print_out)
    on_move, on_click, on_scroll = make_mouse_callbacks(logfile_path, print_out)

    # Start listeners
    kb_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    ms_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)

    kb_listener.start()
    ms_listener.start()

    try:
        # Keep main thread alive while listeners run in background
        while True:
            kb_listener.join(1.0)
            ms_listener.join(1.0)
    except KeyboardInterrupt:
        if print_out:
            print("\nStopping listeners...")
    finally:
        kb_listener.stop()
        ms_listener.stop()
        kb_listener.join()
        ms_listener.join()
        if print_out:
            print("Stopped. Log saved to:", logfile_path.resolve())

if __name__ == "__main__":
    main()
