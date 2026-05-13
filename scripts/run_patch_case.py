from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from structured_diag.actions import list_actions  # noqa: E402
from structured_diag.data import load_run  # noqa: E402
from structured_diag.features import (  # noqa: E402
    build_data_integrity_features,
    build_feature_table,
)
from structured_diag.models import load_cascade  # noqa: E402
from structured_diag.patch_eval import (  # noqa: E402
    PatchCase,
    evaluate_patch,
    write_patch_report,
)
from structured_diag.utils import setup_logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--case-id", default=None)
    p.add_argument("--before", default=None, help="Before run_id (single-case mode).")
    p.add_argument("--after", default=None, help="After run_id (single-case mode).")
    p.add_argument("--action", default=None, help="Action name from the allowlist.")
    p.add_argument(
        "--params", default="{}", help="JSON object of action parameters (single-case mode)."
    )
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="JSON file with a list of patch cases for batch mode.",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory in batch mode.")
    p.add_argument("--list-actions", action="store_true", help="Print the allowlist and exit.")
    p.add_argument("--no-integrity", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _load_cases_file(path: Path) -> list[PatchCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"Expected a JSON list at {path}.")
    out: list[PatchCase] = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            raise SystemExit(f"Case #{i} is not an object.")
        out.append(
            PatchCase(
                case_id=str(c.get("case_id") or f"case_{i:03d}"),
                before_run_id=str(c["before"]),
                after_run_id=str(c["after"]),
                action_name=str(c["action"]),
                action_parameters=dict(c.get("params", {})),
            )
        )
    return out


def _resolve_run_dir(corpus: Path, run_id: str) -> Path | None:
    candidate = corpus / run_id
    return candidate if candidate.is_dir() else None


def _maybe_meta(corpus: Path, run_id: str) -> dict[str, Any] | None:
    rd = _resolve_run_dir(corpus, run_id)
    if rd is None:
        return None
    try:
        return load_run(rd).meta
    except Exception:
        return None


def _print_actions() -> None:
    print("Allowlisted actions:")
    for a in list_actions():
        targets = ", ".join(a.target_classes) or "(any)"
        out = f"  {a.name:25s} targets: {targets}"
        if a.parameters:
            params = ", ".join(p.name for p in a.parameters)
            out += f"   params: {params}"
        print(out)


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    if args.list_actions:
        _print_actions()
        return 0
    cascade = load_cascade(args.artifacts)
    base = build_feature_table(args.corpus)
    integrity_columns = None
    full_df = base.df
    if not args.no_integrity:
        try:
            di = build_data_integrity_features(args.corpus, base_table=base)
            integrity_columns = di.integrity_columns
            full_df = di.df
        except Exception as e:
            print(f"WARNING: integrity features unavailable ({e}).", file=sys.stderr)
    if args.cases is not None:
        cases = _load_cases_file(args.cases)
        if args.out_dir is None:
            args.out_dir = Path("results/patches")
    else:
        if not (args.before and args.after and args.action):
            print(
                "ERROR: provide either --cases <file> or --before/--after/--action.",
                file=sys.stderr,
            )
            return 2
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"ERROR: --params is not valid JSON: {e}", file=sys.stderr)
            return 2
        cases = [
            PatchCase(
                case_id=args.case_id or "case_001",
                before_run_id=args.before,
                after_run_id=args.after,
                action_name=args.action,
                action_parameters=params,
            )
        ]
    n_done = 0
    for case in cases:
        try:
            report = evaluate_patch(
                case=case,
                cascade=cascade,
                feature_table=base,
                full_feature_df=full_df,
                integrity_columns=integrity_columns,
                before_meta=_maybe_meta(args.corpus, case.before_run_id),
                after_meta=_maybe_meta(args.corpus, case.after_run_id),
            )
        except KeyError as e:
            print(f"ERROR [{case.case_id}]: {e}", file=sys.stderr)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"ERROR [{case.case_id}]: {e}", file=sys.stderr)
            continue
        if args.cases is not None or args.out_dir is not None:
            md_target = (args.out_dir / f"{case.case_id}.md") if args.out_dir else None
            json_target = (args.out_dir / f"{case.case_id}.json") if args.out_dir else None
        else:
            md_target = args.out_md
            json_target = args.out_json
        written = write_patch_report(report, md_path=md_target, json_path=json_target)
        print(
            f"[{case.case_id}] outcome=`{report.outcome.status}`  "
            f"before={report.before_diagnosis.final_class}  "
            f"after={report.after_diagnosis.final_class}  "
            f"ΔP(healthy)={report.outcome.delta_p_healthy:+.4f}"
        )
        for kind, p in written.items():
            print(f"   [{kind}] -> {p}")
        n_done += 1
    if n_done == 0:
        print("ERROR: no cases evaluated.", file=sys.stderr)
        return 3
    print(f"\nEvaluated {n_done} patch case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
