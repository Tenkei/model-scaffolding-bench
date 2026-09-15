#!/usr/bin/env python3
"""Capture the active runtime environment beside saved benchmark artifacts.

Usage:
    python scripts/capture_environment.py results/<experiment-name>
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys


def capture(command: list[str]) -> str:
    """Return command output, retaining a useful message when it is unavailable."""
    if shutil.which(command[0]) is None:
        return f"{command[0]} is not available on this machine.\n"

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = completed.stdout + completed.stderr
    if completed.returncode:
        output += f"\nCommand exited with status {completed.returncode}.\n"
    return output.replace(str(Path.home()), "~")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write environment.txt beside saved benchmark result artifacts."
    )
    parser.add_argument("result_dir", type=Path, help="Existing results/<experiment-name> directory")
    args = parser.parse_args()

    if not args.result_dir.is_dir():
        parser.error(f"result directory does not exist: {args.result_dir}")

    output_path = args.result_dir / "environment.txt"
    if output_path.exists():
        parser.error(f"refusing to overwrite existing capture: {output_path}")

    sections = [
        ("nvidia-smi", ["nvidia-smi"], "nvidia-smi"),
        ("Python", [sys.executable, "--version"], "python --version"),
        ("Installed packages", [sys.executable, "-m", "pip", "freeze"], "python -m pip freeze"),
    ]
    lines = [
        f"# Environment capture: {datetime.now(timezone.utc).isoformat()}\n",
    ]
    for title, command, display_command in sections:
        lines.extend((f"\n## {title}\n", f"$ {display_command}\n", capture(command)))

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
