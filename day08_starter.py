"""
BUILDCORED ORCAS — Day 08: PocketAgent
========================================
Run a 3B-parameter LLM on your laptop via ollama.
Build a CLI agent with tools: read files, list
directories, answer questions about your system.

Hardware concept: Edge Inference
Your laptop is the edge device. The model runs entirely
on-device — no cloud, no API keys, no latency tax.
3B params × 4-bit = ~1.5 GB RAM. That's your budget,
just like firmware on a microcontroller with limited SRAM.

PREREQUISITES:
- ollama must be running: `ollama serve` in a separate terminal
- Model must be pulled: `ollama pull qwen2.5:3b`

CONTROLS:
- Type a message → agent responds
- Type 'quit' or 'exit' → stop
"""

import subprocess
import os
import sys
import time
import json
import platform
import shutil
import fnmatch

# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    """Verify ollama is running and model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print("ERROR: ollama is not running.")
            print("Fix: Open another terminal and run: ollama serve")
            sys.exit(1)

        if "qwen2.5:3b" not in result.stdout.lower():
            print("ERROR: qwen2.5:3b model not found.")
            print("Fix: Run: ollama pull qwen2.5:3b")
            sys.exit(1)

        print("✓ ollama is running")
        print("✓ qwen2.5:3b model available")
        return True

    except FileNotFoundError:
        print("ERROR: ollama not installed.")
        print("Fix: Download from https://ollama.com")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: ollama not responding.")
        print("Fix: Restart ollama: ollama serve")
        sys.exit(1)


check_ollama()


# ============================================================
# OLLAMA CHAT FUNCTION
# ============================================================

MODEL = "qwen2.5:3b"


def chat_with_ollama(messages):
    """
    Send messages to ollama and get a response.
    Returns (response_text, tokens_per_second).

    Uses the ollama CLI with JSON output for simplicity —
    no extra Python packages needed.
    """
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    prompt_parts.append("Assistant:")
    full_prompt = "\n".join(prompt_parts)

    start_time = time.time()

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, full_prompt],
            capture_output=True, text=True, timeout=120
        )

        elapsed = time.time() - start_time
        response = result.stdout.strip()

        # Rough token estimate (1 token ≈ 4 chars)
        token_estimate = len(response) / 4
        tps = token_estimate / elapsed if elapsed > 0 else 0

        return response, tps

    except subprocess.TimeoutExpired:
        return "Error: Model timed out. Try a shorter question.", 0
    except Exception as e:
        return f"Error: {e}", 0


# ============================================================
# TOOLS
# ============================================================

def tool_list_directory(path="."):
    """List files and folders in a directory."""
    try:
        items = os.listdir(path)
        dirs  = [f"📁 {item}" for item in items if os.path.isdir(os.path.join(path, item))]
        files = [f"📄 {item}" for item in items if os.path.isfile(os.path.join(path, item))]
        result = f"Contents of '{path}':\n"
        result += "\n".join(sorted(dirs) + sorted(files))
        result += f"\n\n({len(dirs)} folders, {len(files)} files)"
        return result
    except Exception as e:
        return f"Error listing '{path}': {e}"


def tool_read_file(filepath):
    """Read the contents of a text file."""
    try:
        with open(filepath, "r") as f:
            content = f.read(2000)  # Limit to 2000 chars
        truncated = " (truncated)" if len(content) >= 2000 else ""
        return f"Contents of '{filepath}'{truncated}:\n\n{content}"
    except Exception as e:
        return f"Error reading '{filepath}': {e}"


def tool_system_info():
    """Get basic system information."""
    info = {
        "os":         platform.system(),
        "os_version": platform.version(),
        "machine":    platform.machine(),
        "processor":  platform.processor(),
        "python":     platform.python_version(),
        "cwd":        os.getcwd(),
        "user":       os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
    }
    result = "System Information:\n"
    for key, value in info.items():
        result += f"  {key}: {value}\n"
    return result


# ============================================================
# TODO #1 COMPLETED — Two new tools added
# ============================================================

def tool_current_time():
    """Get the current date and time with timezone."""
    from datetime import datetime
    import time as _time
    now = datetime.now()
    tz  = _time.strftime("%Z")
    return (
        f"Current date : {now.strftime('%A, %B %d, %Y')}\n"
        f"Current time : {now.strftime('%H:%M:%S')} ({tz})"
    )


def tool_disk_usage(path="."):
    """Show disk usage (total, used, free) for the given path."""
    try:
        total, used, free = shutil.disk_usage(path)
        def fmt(n):
            for unit in ("B", "KB", "MB", "GB", "TB"):
                if n < 1024:
                    return f"{n:.1f} {unit}"
                n /= 1024
            return f"{n:.1f} PB"

        pct_used = used / total * 100
        bar_len  = 30
        filled   = int(bar_len * pct_used / 100)
        bar      = "█" * filled + "░" * (bar_len - filled)

        return (
            f"Disk usage for '{path}':\n"
            f"  Total : {fmt(total)}\n"
            f"  Used  : {fmt(used)}  ({pct_used:.1f}%)\n"
            f"  Free  : {fmt(free)}\n"
            f"  [{bar}]"
        )
    except Exception as e:
        return f"Error getting disk usage for '{path}': {e}"


def tool_find_files(pattern="."):
    """
    Search for files matching a glob pattern under the current directory.
    Usage: find_files *.py   OR   find_files report
    Matches are returned relative to cwd (max 50 results).
    """
    try:
        # If no wildcard, treat as substring search
        if "*" not in pattern and "?" not in pattern:
            pattern = f"*{pattern}*"

        matches = []
        for root, dirs, files in os.walk("."):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    rel = os.path.relpath(os.path.join(root, filename))
                    matches.append(rel)
                if len(matches) >= 50:
                    break
            if len(matches) >= 50:
                break

        if not matches:
            return f"No files found matching '{pattern}' under '{os.getcwd()}'"

        result = f"Files matching '{pattern}' ({len(matches)} found):\n"
        result += "\n".join(f"  📄 {m}" for m in sorted(matches))
        if len(matches) == 50:
            result += "\n  … (limit reached, refine your pattern)"
        return result
    except Exception as e:
        return f"Error searching for '{pattern}': {e}"


# ============================================================
# TOOL REGISTRY
# ============================================================

AVAILABLE_TOOLS = {
    "list_directory": {
        "function":    tool_list_directory,
        "description": "List files and folders in a directory",
        "usage":       "list_directory [path]",
    },
    "read_file": {
        "function":    tool_read_file,
        "description": "Read the contents of a text file",
        "usage":       "read_file <filepath>",
    },
    "system_info": {
        "function":    tool_system_info,
        "description": "Get system information (OS, Python version, etc)",
        "usage":       "system_info",
    },
    "current_time": {
        "function":    tool_current_time,
        "description": "Get the current date, time, and timezone",
        "usage":       "current_time",
    },
    "disk_usage": {
        "function":    tool_disk_usage,
        "description": "Show total / used / free disk space for a path",
        "usage":       "disk_usage [path]",
    },
    "find_files": {
        "function":    tool_find_files,
        "description": "Search for files by name or glob pattern (e.g. *.py)",
        "usage":       "find_files <pattern>",
    },
}


# ============================================================
# TOOL ROUTING
# ============================================================

def try_parse_tool_call(response):
    """
    Check if the model's response contains a tool call.
    Only the FIRST non-empty line is inspected to avoid false
    positives from mid-paragraph hallucinations.

    Returns (tool_name, argument) or (None, None).
    """
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue  # skip blank lines, check first real line only

        if line.upper().startswith("TOOL:"):
            parts     = line[5:].strip().split(maxsplit=1)
            tool_name = parts[0].lower().strip() if parts else None
            argument  = parts[1].strip() if len(parts) > 1 else None

            if tool_name in AVAILABLE_TOOLS:
                return tool_name, argument

        # First non-empty line did NOT start with TOOL: → not a tool call
        break

    return None, None


def execute_tool(tool_name, argument):
    """Run a tool and return its output."""
    func = AVAILABLE_TOOLS[tool_name]["function"]
    try:
        return func(argument) if argument else func()
    except Exception as e:
        return f"Tool error: {e}"


# ============================================================
# TODO #2 COMPLETED — Improved system prompt
# ============================================================

tools_description = "\n".join(
    f"  - {name}: {info['description']}\n    syntax: {info['usage']}"
    for name, info in AVAILABLE_TOOLS.items()
)

SYSTEM_PROMPT = f"""You are PocketAgent, a precise local AI assistant running entirely on this device.
You have access to the following tools:

{tools_description}

STRICT RULES you must follow every response:
1. If the user's request requires a tool, your ENTIRE response must be ONE line:
   TOOL: <tool_name> [argument]
   Examples:
     TOOL: list_directory /home
     TOOL: read_file README.md
     TOOL: system_info
     TOOL: current_time
     TOOL: disk_usage /
     TOOL: find_files *.py
2. Do NOT add any explanation before or after the TOOL: line.
3. Do NOT call more than one tool per turn.
4. If you can answer without a tool (math, general knowledge, conversation), reply normally in plain text.
5. Keep answers short and factual — you run on limited compute.
6. Never invent file contents or system data. Use tools to get real information."""


# ============================================================
# MAIN CHAT LOOP
# ============================================================

def print_header():
    print()
    print("=" * 58)
    print("  🤖 PocketAgent — Local AI Assistant")
    print(f"  Model: {MODEL} | Running on: {platform.system()}")
    print("=" * 58)
    print()
    print("  Available tools:")
    for name, info in AVAILABLE_TOOLS.items():
        print(f"    • {name:<16} — {info['description']}")
    print()
    print("  Type a question or command. Type 'quit' to exit.")
    print("  Try: 'What files are in this directory?'")
    print("       'How much disk space do I have?'")
    print("       'Find all Python files here'")
    print("       'What time is it?'")
    print()


def main():
    print_header()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        print("\n⏳ Thinking...", end="", flush=True)
        response, tps = chat_with_ollama(messages)
        print("\r                    \r", end="")

        tool_name, argument = try_parse_tool_call(response)

        if tool_name:
            print(f"🔧 Using tool: {tool_name}", end="")
            if argument:
                print(f" ({argument})")
            else:
                print()

            tool_output = execute_tool(tool_name, argument)
            print(f"\n{tool_output}\n")

            # Feed result back so model can explain it
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Tool result:\n{tool_output}\n\nBriefly explain what this shows."
            })

            print("⏳ Analyzing...", end="", flush=True)
            explanation, tps = chat_with_ollama(messages)
            print("\r                    \r", end="")

            print(f"Agent > {explanation}")
            messages.append({"role": "assistant", "content": explanation})

        else:
            print(f"Agent > {response}")
            messages.append({"role": "assistant", "content": response})

        print(f"\n  ⚡ {tps:.1f} tokens/sec\n")

        # Keep history manageable: system prompt + last 10 messages
        if len(messages) > 12:
            messages = [messages[0]] + messages[-10:]


if __name__ == "__main__":
    main()
