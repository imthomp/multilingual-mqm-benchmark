#!/usr/bin/env python3
"""CLI entry point for the multilingual MQM benchmark pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --settings path/to/settings.toml
    python scripts/run_pipeline.py --lang es pt  # run only specific languages
"""

import argparse
import sys
from pathlib import Path

# Make sure the package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mqmbench.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MT metrics against human MQM annotations across 12 languages."
    )
    parser.add_argument(
        "--settings",
        default="settings.toml",
        help="Path to settings TOML file (default: settings.toml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settings_path = Path(args.settings).resolve()
    if not settings_path.exists():
        print(f"Error: settings file not found: {settings_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Using settings: {settings_path}")
    results = run_pipeline(str(settings_path))

    print("\n=== Correlation Summary by Resource Tier ===")
    print(results["tier_summary"].to_string(index=False))

    print("\n=== Per-Language Results ===")
    print(results["correlations"].to_string(index=False))


if __name__ == "__main__":
    main()
