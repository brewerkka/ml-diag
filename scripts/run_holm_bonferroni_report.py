from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from structured_diag.evaluation import holm_bonferroni_adjust  # noqa: E402

PRIMARY: dict = {
    "name": "stacking_gbm accuracy vs flat",
    "corpus": "real_8ds_n5_multi",
    "n_test": 160,
    "delta_point": 0.0312,
    "p_b_better": 0.820,
    "cohen_h_approx": 0.084,
    "magnitude": "trivial",
    "reporting_mode": "marginal-only (primary endpoint)",
}

SECONDARY_FAMILY: list[dict] = [
    {
        "name": "LLM arbitration macro-F1 vs flat",
        "corpus": "real_8ds_n5_multi",
        "n_test": 160,
        "delta_point": 0.0256,
        "p_b_better": 0.993,
        "cohen_h_approx": None,
        "magnitude": "n/a",
    },
    {
        "name": "LLM arbitration accuracy vs flat",
        "corpus": "real_8ds_n5_multi",
        "n_test": 160,
        "delta_point": 0.0250,
        "p_b_better": 0.985,
        "cohen_h_approx": 0.067,
        "magnitude": "trivial",
    },
    {
        "name": "stacking_gbm accuracy vs cascade",
        "corpus": "real_8ds_n5_multi",
        "n_test": 160,
        "delta_point": 0.0500,
        "p_b_better": 0.956,
        "cohen_h_approx": 0.130,
        "magnitude": "trivial",
    },
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out-md", default="results/holm_bonferroni_report.md", type=Path)
    p.add_argument("--out-json", default="results/holm_bonferroni_report.json", type=Path)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    p_values = [s["p_b_better"] for s in SECONDARY_FAMILY]
    adjusted = holm_bonferroni_adjust(p_values, alpha=args.alpha)
    for s, a in zip(SECONDARY_FAMILY, adjusted):
        s["holm_rank"] = a["rank"]
        s["holm_threshold_p_better"] = a["holm_threshold_p_better"]
        s["holm_significant"] = a["significant"]
    lines = [
        "# Holm-Bonferroni-adjusted family-wise significance",
        "",
        f"Family-wise α = {args.alpha}; secondary endpoints m = {len(SECONDARY_FAMILY)}.",
        "Method: Holm (1979) sequential step-down rejection.",
        "",
        "## Primary endpoint (NOT in Holm family — reported marginally)",
        "",
        f"* **{PRIMARY['name']}** on `{PRIMARY['corpus']}` (n_test = {PRIMARY['n_test']}):",
        f"  Δ = +{PRIMARY['delta_point']:.4f}, marginal P_better = {PRIMARY['p_b_better']:.3f}, "
        f"Cohen's h ≈ {PRIMARY['cohen_h_approx']:.3f} ({PRIMARY['magnitude']}).",
        "  Reported as the system-level headline; **not** subject to Holm "
        "correction by pre-registration.",
        "",
        "## Secondary family (Holm-corrected at α/m sequential thresholds)",
        "",
        "| # | Comparison | P_better | Cohen's h | Magnitude | Holm rank | Holm threshold P_better | Significant after Holm |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(SECONDARY_FAMILY, start=1):
        h = "—" if s["cohen_h_approx"] is None else f"{s['cohen_h_approx']:.3f}"
        sig = "✓" if s["holm_significant"] else "✗"
        lines.append(
            f"| {i} | {s['name']} | {s['p_b_better']:.4f} | {h} | "
            f"{s['magnitude']} | {s['holm_rank']} | "
            f"{s['holm_threshold_p_better']:.4f} | **{sig}** |"
        )
    lines.extend(
        [
            "",
            "## Interpretation for the thesis",
            "",
            "All three secondary comparisons pass family-wise α = 0.05 under "
            "Holm-Bonferroni: the strongest signal (LLM arbitration macro-F1) "
            "needs to exceed 0.9833, which it does (0.993); the next "
            "(LLM arbitration accuracy) needs 0.9750 and reaches 0.985; "
            "the third (stacking vs cascade) needs 0.9500 and reaches 0.956. "
            "The primary endpoint (stacking vs flat, P_better = 0.820) is "
            "reported marginally per pre-registration; it does **not** "
            "survive a hypothetical family-of-4 Holm correction "
            "(threshold for rank 4 would be 0.9500).",
            "",
            "Cohen's h effect sizes for accuracy gains (~0.07–0.13) are "
            "*trivial* by Cohen (1988) thresholds (0.2 small / 0.5 medium / "
            "0.8 large). This is the normal pattern for moderate-n ML "
            "benchmarks: statistically reliable improvements of small "
            "practical magnitude. Reporting both metrics (P_better + Cohen's "
            "h) lets the committee distinguish 'reliable' from 'large'.",
            "",
        ]
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    payload = {
        "method": "Holm (1979) sequential Bonferroni",
        "alpha_family_wise": args.alpha,
        "primary_endpoint": PRIMARY,
        "secondary_family_size": len(SECONDARY_FAMILY),
        "secondary_family": SECONDARY_FAMILY,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote JSON     -> {args.out_json}")
    print()
    print("Holm-Bonferroni report summary:")
    print(
        f"  Primary  ({PRIMARY['name']}): P_better = {PRIMARY['p_b_better']:.3f} "
        f"(marginal, not Holm-corrected)"
    )
    print(f"  Secondary family of m={len(SECONDARY_FAMILY)} (α = {args.alpha}):")
    for s in SECONDARY_FAMILY:
        sig = "SIGNIFICANT" if s["holm_significant"] else "NOT SIGNIFICANT"
        print(
            f"    {s['name']:50s} P={s['p_b_better']:.3f}  "
            f"rank={s['holm_rank']}  threshold={s['holm_threshold_p_better']:.4f}  "
            f"=> {sig}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
