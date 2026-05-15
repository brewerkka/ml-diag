from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_diag.benchmark import (              
    PartitionResult,
    partition_corpus,
    rules_from_mapping,
)
from ml_diag.utils import setup_logging              


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path, help="Path to corpus directory.")
    p.add_argument("--out", required=True, type=Path, help="Where to write the JSON summary.")
    p.add_argument(
        "--config", type=Path, default=None, help="Optional YAML with a `partition:` section."
    )
    p.add_argument("--skip-broken", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _load_rules_from_config(config_path: Path | None) -> Any:
    if config_path is None:
        return None
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Config root must be a mapping: {config_path}")
    return payload.get("partition")


def _print_summary(result: PartitionResult) -> None:
    summary = result.to_summary_dict()
    print("=" * 72)
    print(f"Corpus: {result.corpus_name}")
    print(f"Path:   {result.corpus_path}")
    print(f"Rules:  {json.dumps(summary['rules'], sort_keys=True)}")
    print("-" * 72)
    counts = summary["counts"]
    total = max(1, counts["total"])
    print(
        f"Total runs: {counts['total']} | "
        f"core: {counts['core']} ({100.0 * counts['core'] / total:.1f}%) | "
        f"extended: {counts['extended']} ({100.0 * counts['extended'] / total:.1f}%)"
    )
    print("-" * 72)
    for slice_name in ("core", "extended"):
        bal = summary["class_balance"][slice_name]
        sub_total = sum(bal.values()) or 1
        print(f"\n{slice_name.upper()} class balance:")
        if not bal:
            print("  (empty)")
            continue
        for label, n in sorted(bal.items(), key=lambda kv: -kv[1]):
            shown = label if label is not None else "<unlabeled>"
            print(f"  {shown:30s} {n:5d}  ({100.0 * n / sub_total:5.1f}%)")
    print("\nReason counts (why runs were pushed to extended):")
    if not summary["reason_counts"]:
        print("  (no reasons recorded — this should not happen if extended is non-empty)")
    else:
        for reason, n in summary["reason_counts"].items():
            print(f"  {reason:35s} {n:5d}")
    print("=" * 72)


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    rules_payload = _load_rules_from_config(args.config)
    rules = rules_from_mapping(rules_payload)
    result = partition_corpus(args.corpus, rules=rules, skip_broken=args.skip_broken)
    out_path = result.save_summary(args.out)
    _print_summary(result)
    print(f"\nWrote summary -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
