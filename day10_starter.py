"""
BUILDCORED ORCAS — Day 10: TerminalBrain
==========================================
TODO #1 ✅  Improved error detection patterns
TODO #2 ✅  Tuned LLM prompt
TODO #3 ✅  Smart pattern caching with normalised keys
"""

import sys
import os
import platform
import subprocess
import threading
import queue
import time
import re
import argparse

IS_WINDOWS = platform.system() == "Windows"

if not IS_WINDOWS:
    try:
        import pty
        import select
        HAS_PTY = True
    except ImportError:
        HAS_PTY = False
else:
    HAS_PTY = False


class Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"


def color_text(text, color):
    return f"{color}{text}{Color.RESET}"


MODEL = "qwen2.5:3b"


def check_ollama():
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return False, "ollama not running. Run: ollama serve"
        if "qwen2.5" not in result.stdout.lower():
            return False, "Model missing. Run: ollama pull qwen2.5:3b"
        return True, "ok"
    except FileNotFoundError:
        return False, "ollama not installed. Get it from https://ollama.com"
    except Exception as e:
        return False, str(e)


def ask_llm_for_fix(error_text):
    prompt = build_llm_prompt(error_text)
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[LLM timeout — error too complex]"
    except Exception as e:
        return f"[LLM error: {e}]"


# ============================================================
# TODO #2 ✅ — Tuned LLM prompt
# Key improvements:
#   • Injects the host OS so commands are platform-correct
#   • Bans greetings, explanations, bullet lists
#   • Demands EXACT commands, not paraphrases
#   • Uses few-shot examples to lock in the output style
# ============================================================

def build_llm_prompt(error_text):
    os_name = platform.system()
    return f"""You are a terminal error fixer running on {os_name}.
A command just produced this error:

{error_text}

Rules:
- ONE fix only, maximum 2 sentences.
- If a shell command fixes it, write the EXACT command (e.g. `pip install requests`).
- If a code change fixes it, show only the corrected line.
- Do NOT greet, explain the error, or list multiple options.
- Do NOT say "try" — state the fix directly.

Examples of good replies:
  "Run: pip install numpy"
  "Change line 5 to: import os.path instead of import os"
  "Run: chmod +x script.sh"

Your fix:"""


# ============================================================
# TODO #1 ✅ — Expanded error detection patterns
# Added: extra Python exceptions, shell/FS errors, network,
# pip/npm, Node.js, compilers (gcc/rustc), git, docker,
# Windows-specific messages, and generic severity keywords.
# ============================================================

ERROR_PATTERNS = [
    # Python built-in exceptions
    r"Traceback \(most recent call last\)",
    r"Error:", r"Exception:",
    r"ModuleNotFoundError", r"ImportError",
    r"NameError", r"SyntaxError", r"TypeError",
    r"ValueError", r"KeyError", r"AttributeError",
    r"FileNotFoundError", r"IndexError", r"RuntimeError",
    r"RecursionError", r"OverflowError", r"MemoryError",
    r"ZeroDivisionError", r"IndentationError",
    r"UnicodeDecodeError", r"UnicodeEncodeError",
    r"PermissionError", r"TimeoutError", r"ConnectionError",
    r"AssertionError", r"NotImplementedError",
    r"OSError", r"IOError", r"StopIteration",

    # Shell / filesystem
    r"command not found",
    r"No such file or directory",
    r"permission denied",
    r"cannot access",
    r"is not recognized as",       # Windows
    r"Access is denied",           # Windows
    r"not found",
    r"cannot open", r"failed to open",
    r"is a directory",
    r"too many open files",
    r"read-only file system",
    r"disk quota exceeded",
    r"no space left on device",
    r"operation not permitted",

    # Network / TLS
    r"connection refused",
    r"connection timed out",
    r"name or service not known",
    r"network unreachable",
    r"could not resolve host",
    r"ssl.*error",
    r"certificate verify failed",

    # pip / package managers
    r"could not find a version",
    r"no matching distribution",
    r"requirement.*not satisfied",
    r"ERROR: pip",
    r"npm ERR!",
    r"yarn error",
    r"package not found",

    # Node.js / JavaScript
    r"cannot find module",
    r"ReferenceError",
    r"uncaughtException",
    r"unhandledPromiseRejectionWarning",

    # Compilers (gcc / clang / rustc / make)
    r"error\[E\d+\]",              # Rust
    r"error: ld returned",         # linker
    r"undefined reference to",
    r"undefined symbol",
    r"compilation failed",
    r"make\[.*\]: \*\*\* ",        # make

    # Git
    r"fatal: ",
    r"error: failed to push",
    r"merge conflict",
    r"not a git repository",

    # Docker
    r"docker: error",
    r"container.*exited with",
    r"image.*not found",
    r"bind.*address already in use",

    # Generic severity
    r"FAILED", r"FATAL", r"CRITICAL", r"ABORT",
    r"Segmentation fault",
    r"core dumped",
    r"killed",
    r"out of memory",
]

ERROR_REGEX = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)


def is_error_line(line):
    if not line or not line.strip():
        return False
    return bool(ERROR_REGEX.search(line))


# ============================================================
# TODO #3 ✅ — Smart pattern caching
# Improvements over the bare scaffold:
#   • _make_cache_key() extracts the core error keyword +
#     surrounding context so the same logical error maps to
#     the same key even when tracebacks differ.
#   • handle_error_block() returns (used_cache, fix) so
#     run_with_brain() can update llm_calls / cache_hits.
# ============================================================

fix_cache = {}

_SIG_RE = re.compile(
    r"(ModuleNotFoundError|ImportError|NameError|SyntaxError|TypeError|"
    r"ValueError|KeyError|AttributeError|FileNotFoundError|RuntimeError|"
    r"IndexError|ZeroDivisionError|IndentationError|PermissionError|"
    r"ConnectionError|TimeoutError|OSError|Error|Exception|fatal|FATAL|"
    r"FAILED|command not found|No such file or directory)[^\n]*",
    re.IGNORECASE,
)


def _make_cache_key(error_text):
    match = _SIG_RE.search(error_text)
    if match:
        return re.sub(r"\s+", " ", match.group(0)).strip().lower()
    return error_text[:80].strip().lower()


def get_cached_fix(error_text):
    return fix_cache.get(_make_cache_key(error_text))


def cache_fix(error_text, fix):
    fix_cache[_make_cache_key(error_text)] = fix


# ============================================================
# THREADED I/O
# ============================================================

def reader_thread(stream, output_queue, stream_name):
    try:
        for line in iter(stream.readline, ''):
            if not line:
                break
            output_queue.put((stream_name, line))
    except Exception as e:
        output_queue.put(("error", f"[reader thread error: {e}]\n"))
    finally:
        try:
            stream.close()
        except Exception:
            pass


# ============================================================
# MAIN COMMAND WRAPPER
# ============================================================

def run_with_brain(command):
    print(color_text(f"\n┌─ TerminalBrain wrapping: ", Color.CYAN), end="")
    print(color_text(" ".join(command), Color.BOLD))
    print(color_text("│ stdout = white | stderr = red | brain = cyan", Color.DIM))
    print(color_text("└─" + "─" * 50, Color.CYAN))
    print()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except FileNotFoundError:
        print(color_text(f"Command not found: {command[0]}", Color.RED))
        suggestion = ask_llm_for_fix(f"command not found: {command[0]}")
        print(color_text(f"\n🧠 Brain: {suggestion}\n", Color.CYAN))
        return
    except Exception as e:
        print(color_text(f"Failed to start process: {e}", Color.RED))
        return

    output_queue = queue.Queue()

    stdout_thread = threading.Thread(
        target=reader_thread,
        args=(process.stdout, output_queue, "stdout"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=reader_thread,
        args=(process.stderr, output_queue, "stderr"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    error_buffer = []
    in_error_block = False
    error_count = 0
    llm_calls = 0   # TODO #3 ✅ — now properly counted
    cache_hits = 0  # TODO #3 ✅ — now properly counted

    while True:
        try:
            stream_name, line = output_queue.get(timeout=0.1)
        except queue.Empty:
            if process.poll() is not None and output_queue.empty():
                break
            continue

        if stream_name == "stdout":
            print(color_text(line.rstrip(), Color.WHITE))
            if in_error_block and error_buffer:
                used_cache, _ = handle_error_block(error_buffer)
                if used_cache:
                    cache_hits += 1
                else:
                    llm_calls += 1
                error_buffer = []
                in_error_block = False

        elif stream_name == "stderr":
            print(color_text(line.rstrip(), Color.RED))
            if is_error_line(line):
                in_error_block = True
                error_count += 1
            if in_error_block:
                error_buffer.append(line)

    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)

    if error_buffer:
        used_cache, _ = handle_error_block(error_buffer)
        if used_cache:
            cache_hits += 1
        else:
            llm_calls += 1

    print()
    print(color_text("─" * 52, Color.DIM))
    exit_code = process.returncode
    status_color = Color.GREEN if exit_code == 0 else Color.RED
    print(color_text(f"Exit code: {exit_code}", status_color))
    print(color_text(f"Errors detected: {error_count}", Color.DIM))
    print(color_text(f"LLM calls: {llm_calls} | cache hits: {cache_hits}", Color.DIM))


def handle_error_block(lines):
    """Returns (used_cache: bool, fix: str)."""
    error_text = "".join(lines).strip()
    if not error_text:
        return False, ""

    cached = get_cached_fix(error_text)
    if cached:
        print()
        print(color_text(f"🧠 Brain (cached): {cached}", Color.CYAN))
        print()
        return True, cached

    print()
    print(color_text("🧠 Brain analyzing...", Color.CYAN), end="", flush=True)
    fix = ask_llm_for_fix(error_text)
    print("\r" + " " * 30 + "\r", end="")
    print(color_text(f"🧠 Brain: {fix}", Color.CYAN))
    print()

    cache_fix(error_text, fix)
    return False, fix


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TerminalBrain — wrap a command and analyze its errors with a local LLM"
    )
    parser.add_argument("command", nargs="+", help="The command to run")
    args = parser.parse_args()

    ok, msg = check_ollama()
    if not ok:
        print(color_text(f"ERROR: {msg}", Color.RED))
        sys.exit(1)

    print(color_text("✓ ollama ready", Color.GREEN))
    print(color_text(f"  Model: {MODEL}", Color.DIM))
    print(color_text(f"  Platform: {platform.system()}", Color.DIM))
    print(color_text(f"  pty available: {HAS_PTY}", Color.DIM))

    run_with_brain(args.command)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print()
        print("TerminalBrain — wrap a command and get AI fix suggestions")
        print()
        print("Usage:")
        print("  python day10_starter.py <command> [args...]")
        print()
        print("Try one of these to test:")
        print('  python day10_starter.py python -c "import nonexistent_module"')
        print("  python day10_starter.py ls /nonexistent_directory")
        print('  python day10_starter.py python -c "print(undefined_var)"')
        print()
        sys.exit(0)

    main()
