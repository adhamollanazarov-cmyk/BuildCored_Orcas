"""
BUILDCORED ORCAS — Day 13: DailyDebrief
Collect your day's activity and get an AI summary.

Hardware concept: Flight Data Recorder
Collect all streams → compress → report.

TASKS:
1. Tune the debrief prompt (TODO #1)
2. Add a 4th data source (TODO #2)

Run: python day13_starter.py
"""
import subprocess, os, sys, time, json
from pathlib import Path
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except ImportError:
    print("pip install rich"); sys.exit(1)

MODEL = "qwen2.5:3b"

def check_ollama():
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if "qwen2.5" not in r.stdout.lower():
            console.print("[red]Run: ollama pull qwen2.5:3b[/red]"); sys.exit(1)
    except:
        console.print("[red]ollama not found[/red]"); sys.exit(1)

check_ollama()

# ====== DATA SOURCES ======

def get_git_commits(hours=24):
    """Last N hours of git commits from current repo."""
    try:
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        r = subprocess.run(
            ["git", "log", f"--since={since}", "--pretty=format:%h %s"],
            capture_output=True, text=True, timeout=5
        )
        commits = r.stdout.strip().split("\n") if r.stdout.strip() else []
        return commits[:20]
    except:
        return []

def get_recent_files(hours=24):
    """Files modified in the last N hours in home dir."""
    home = Path.home()
    cutoff = time.time() - (hours * 3600)
    recent = []
    skip_parts = [".cache", "node_modules", "__pycache__", ".git", "Library"]

    for p in home.rglob("*"):
        try:
            if p.is_file() and p.stat().st_mtime > cutoff:
                if any(part in str(p) for part in skip_parts):
                    continue
                recent.append(str(p.relative_to(home)))
                if len(recent) >= 30:
                    break
        except:
            pass
    return recent

def get_shell_history(lines=30):
    """Last N lines of shell history."""
    for hist_file in [".zsh_history", ".bash_history"]:
        path = Path.home() / hist_file
        if path.exists():
            try:
                with open(path, "r", errors="ignore") as f:
                    all_lines = f.readlines()
                return [l.strip() for l in all_lines[-lines:] if l.strip()]
            except:
                pass
    return []

def get_vscode_recent(hours=24):
    """Recent VS Code workspaces/files from storage JSON."""
    candidates = [
        Path.home() / "AppData/Roaming/Code/User/globalStorage/storage.json",            # Windows
        Path.home() / "Library/Application Support/Code/User/globalStorage/storage.json", # macOS
        Path.home() / ".config/Code/User/globalStorage/storage.json",                     # Linux
    ]

    cutoff = time.time() - (hours * 3600)
    recent = []

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                continue

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)

            for key, value in data.items():
                key_lower = str(key).lower()

                if "recent" in key_lower or "history" in key_lower or "opened" in key_lower:
                    if isinstance(value, list):
                        for item in value:
                            recent.append(str(item))
                    elif isinstance(value, dict):
                        for subk, subv in value.items():
                            recent.append(f"{subk}: {subv}")
                    else:
                        recent.append(f"{key}: {value}")

            cleaned = []
            seen = set()
            for item in recent:
                item = item.strip()
                if not item:
                    continue
                if item not in seen:
                    seen.add(item)
                    cleaned.append(item)
                if len(cleaned) >= 20:
                    break

            return cleaned
        except:
            pass

    return []

# ====== LLM SUMMARY ======

DEBRIEF_PROMPT = """You are summarizing a developer's last 24 hours from noisy activity logs.

Rules:
- Infer carefully from the evidence only
- Be concrete and concise
- Do not mention missing data
- If something is uncertain, use cautious wording like "likely" or "seems"
- Output EXACTLY these 5 lines and nothing else
- Each line must be short, useful, and based on the data

Format exactly:
BUILT: ...
BROKE: ...
LEARNED: ...
PATTERN: ...
NEXT: ...

Data:
{data}
"""

def get_debrief(data_text):
    prompt = DEBRIEF_PROMPT.format(data=data_text[:3000])
    try:
        r = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True, text=True, timeout=60
        )
        return r.stdout.strip()
    except:
        return "[LLM error]"

# ====== MAIN ======

console.print("\n[bold cyan]📊 DailyDebrief[/bold cyan]\n")
console.print("[dim]Collecting data from the last 24 hours...[/dim]\n")

commits = get_git_commits()
files = get_recent_files()
history = get_shell_history()
vscode = get_vscode_recent()

console.print(f"  Git commits:    {len(commits)}")
console.print(f"  Recent files:   {len(files)}")
console.print(f"  Shell commands: {len(history)}")
console.print(f"  VS Code recent: {len(vscode)}")
console.print()

# Build data string for LLM
data = []
if commits:
    data.append("GIT COMMITS:\n" + "\n".join(commits[:10]))
if files:
    data.append("FILES MODIFIED:\n" + "\n".join(files[:15]))
if history:
    data.append("SHELL HISTORY:\n" + "\n".join(history[-15:]))
if vscode:
    data.append("VS CODE RECENT:\n" + "\n".join(vscode[:10]))

if not data:
    console.print("[yellow]No data found. Make some activity first![/yellow]")
    sys.exit(0)

combined = "\n\n".join(data)

console.print("[dim]Asking the brain...[/dim]")
start = time.time()
debrief = get_debrief(combined)
elapsed = time.time() - start

console.print(Panel(
    debrief,
    title=f"Today's Debrief ({elapsed:.1f}s)",
    border_style="cyan"
))
console.print("\nSee you tomorrow for Day 14!")
