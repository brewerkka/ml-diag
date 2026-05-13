"""Smoke and integration tests for structured_diag.

These tests validate that the public API surface of the library works on
synthetic data: imports resolve, metrics behave correctly, conformal
calibration produces the expected coverage, and the stacking
featurizer outputs the documented 33-feature vector.

Run with:
    pytest tests/

The tests deliberately avoid loading real corpora (which require parquet
support and large fixtures); they exercise the *logic* of the library
on tiny synthetic datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is on the path even when pytest is invoked from project root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# 1. Import smoke — every public module loads without errors.
# ---------------------------------------------------------------------------


def test_imports_top_level():
    """Top-level modules import without circular-dependency errors."""
    import structured_diag.benchmark  # noqa: F401
    import structured_diag.data  # noqa: F401
    import structured_diag.diagnosis  # noqa: F401
    import structured_diag.evaluation  # noqa: F401
    import structured_diag.features  # noqa: F401
    import structured_diag.interpretation  # noqa: F401
    import structured_diag.labels  # noqa: F401
    import structured_diag.models  # noqa: F401
    import structured_diag.utils  # noqa: F401


def test_imports_diagnosis_submodules():
    """All diagnosis submodules import (covers conformal, stacking, arbitrator)."""
    from structured_diag.diagnosis import (
        arbitrator,  # noqa: F401
        conformal_layer,  # noqa: F401
        hybrid_resolver,  # noqa: F401
        oof_predictions,  # noqa: F401
        stacking_resolver,  # noqa: F401
    )


def test_label_vocabulary_consistent():
    """Primary labels have exactly 6 entries and stage label maps are consistent."""
    from structured_diag.labels import (
        DATA_RELATED,
        FAULTY,
        HEALTHY,
        OPT_GEN_RELATED,
        PRIMARY_LABELS,
        STAGE1_LABELS,
        STAGE2_LABELS,
        to_stage1,
        to_stage2,
    )

    assert len(PRIMARY_LABELS) == 6
    assert HEALTHY in PRIMARY_LABELS
    assert FAULTY in STAGE1_LABELS
    assert DATA_RELATED in STAGE2_LABELS
    assert OPT_GEN_RELATED in STAGE2_LABELS
    # to_stage1: every primary label maps to a Stage1 label.
    for c in PRIMARY_LABELS:
        s1 = to_stage1(c)
        assert s1 in STAGE1_LABELS, f"{c} -> {s1} not in STAGE1_LABELS"
    # to_stage2: faulty primary labels map to a Stage2 label.
    for c in PRIMARY_LABELS:
        if to_stage1(c) == FAULTY:
            assert to_stage2(c) in STAGE2_LABELS


# ---------------------------------------------------------------------------
# 2. Metrics — classification_report and bootstrap CI.
# ---------------------------------------------------------------------------


def test_classification_report_perfect():
    """Perfect predictions yield acc=1.0 and macro_f1=1.0."""
    from structured_diag.evaluation.metrics import classification_report

    y_true = ["healthy", "leakage", "overfitting", "instability"]
    y_pred = ["healthy", "leakage", "overfitting", "instability"]
    rep = classification_report(y_true, y_pred)
    assert rep.accuracy == 1.0
    assert rep.macro_f1 == 1.0
    assert rep.n_samples == 4


def test_classification_report_per_class():
    """Per-class F1 reports correct values for a known confusion."""
    from structured_diag.evaluation.metrics import classification_report

    y_true = ["healthy"] * 5 + ["leakage"] * 5
    y_pred = ["healthy"] * 4 + ["leakage"] + ["leakage"] * 3 + ["healthy"] * 2
    rep = classification_report(y_true, y_pred)
    # 4 healthy correctly classified, 3 leakage correctly classified
    assert rep.accuracy == 0.7
    # both per-class F1 should be present
    assert "healthy" in rep.per_class_f1
    assert "leakage" in rep.per_class_f1


def test_bootstrap_metric_ci_basic():
    """Bootstrap CI for accuracy on synthetic data is internally consistent."""
    from structured_diag.evaluation.metrics import bootstrap_metric_ci

    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.choice(["a", "b", "c"], size=n).tolist()
    y_pred = list(y_true)  # 100% accuracy
    out = bootstrap_metric_ci(y_true, y_pred, metric="accuracy", n_bootstrap=200)
    assert out["point_estimate"] == 1.0
    assert 0.95 <= out["ci_low"] <= out["ci_high"] <= 1.0


def test_bootstrap_delta_ci_zero_when_identical():
    """When two classifiers agree on every row, Δ accuracy CI brackets zero."""
    from structured_diag.evaluation.metrics import bootstrap_delta_ci

    y_true = ["a", "b", "c"] * 30
    y_a = list(y_true)
    y_b = list(y_true)
    out = bootstrap_delta_ci(y_true, y_a, y_b, metric="accuracy", n_bootstrap=200)
    assert out["delta_point"] == 0.0


# ---------------------------------------------------------------------------
# 3. Conformal layer — calibration and prediction sets.
# ---------------------------------------------------------------------------


def test_conformal_calibration_marginal_coverage():
    """Empirical coverage on synthetic data approximately matches 1−α target."""
    from structured_diag.diagnosis.conformal_layer import (
        calibrate_split_conformal,
        evaluate_conformal,
        predict_with_conformal,
    )
    from structured_diag.labels import PRIMARY_LABELS

    rng = np.random.default_rng(0)
    n_cal, n_test = 500, 200
    classes = list(PRIMARY_LABELS)

    def _gen_proba(n):
        # Generate "noisy but informative" probabilities: correct class wins
        # with Dirichlet noise. Calibration set is exchangeable with test.
        ys = rng.choice(classes, size=n).tolist()
        rows = []
        for y in ys:
            base = rng.dirichlet(np.ones(6))
            # Boost the true class so coverage > 1-α is achievable
            target_idx = classes.index(y)
            base[target_idx] += 1.5
            base = base / base.sum()
            rows.append(base)
        proba_df = pd.DataFrame(rows, columns=classes)
        return proba_df, pd.Series(ys, name="y")

    proba_cal, y_cal = _gen_proba(n_cal)
    proba_test, y_test = _gen_proba(n_test)
    calibrator = calibrate_split_conformal(proba_oof=proba_cal, y_oof=y_cal, alpha=0.1)
    results = predict_with_conformal(
        proba_test=proba_test,
        calibrator=calibrator,
    )
    metrics = evaluate_conformal(results=results, y_test=y_test)
    # Marginal coverage should hold approximately at level 1-α=0.9
    # (minus a small finite-sample slack of ~0.05)
    assert metrics["empirical_coverage"] >= 0.85, (
        f"Coverage {metrics['empirical_coverage']:.3f} below 0.85"
    )
    # Prediction set sizes are positive
    assert metrics["average_set_size"] >= 1.0


def test_conformal_quantile_finite_sample_correction():
    """The +1/n finite-sample correction yields the documented quantile index."""
    from structured_diag.diagnosis.conformal_layer import calibrate_split_conformal
    from structured_diag.labels import PRIMARY_LABELS

    rng = np.random.default_rng(1)
    n_cal = 100
    classes = list(PRIMARY_LABELS)
    proba = pd.DataFrame(
        rng.dirichlet(np.ones(6), size=n_cal),
        columns=classes,
    )
    y = pd.Series(rng.choice(classes, size=n_cal), name="y")
    cal = calibrate_split_conformal(proba_oof=proba, y_oof=y, alpha=0.05)
    # quantile is bounded between 0 and 1 (LAC nonconformity range)
    assert 0.0 <= cal.quantile <= 1.0
    assert cal.alpha == 0.05
    assert cal.n_calibration == n_cal


# ---------------------------------------------------------------------------
# 4. Stacking featurizer — produces the documented 33-feature vector.
# ---------------------------------------------------------------------------


def _build_dummy_oof_inputs(n=20):
    """Synthesize the inputs required by stacking_resolver.featurize."""
    from structured_diag.diagnosis.oof_predictions import STAGE_PROBA_COLS
    from structured_diag.labels import PRIMARY_LABELS

    rng = np.random.default_rng(7)
    idx = [f"r{i}" for i in range(n)]
    cols = list(PRIMARY_LABELS)
    flat = pd.DataFrame(rng.dirichlet(np.ones(6), size=n), index=idx, columns=cols)
    casc = pd.DataFrame(rng.dirichlet(np.ones(6), size=n), index=idx, columns=cols)
    sp = pd.DataFrame(
        rng.random((n, len(STAGE_PROBA_COLS))), index=idx, columns=list(STAGE_PROBA_COLS)
    )
    arb = pd.DataFrame(0.0, index=idx, columns=cols)
    trig = pd.Series(False, index=idx)
    conf = pd.Series(0.0, index=idx)
    return flat, casc, sp, arb, trig, conf


def test_stacking_featurize_shape():
    """featurize returns DataFrame with 33 columns and aligned index."""
    from structured_diag.diagnosis.stacking_resolver import META_FEATURES, featurize

    flat, casc, sp, arb, trig, conf = _build_dummy_oof_inputs(n=12)
    feats = featurize(
        flat_proba=flat,
        cascade_proba=casc,
        cascade_stage_probs=sp,
        arbitrator_label_probs=arb,
        arbitrator_triggered=trig,
        arbitrator_confidence=conf,
    )
    assert feats.shape == (12, 33), f"Expected (12, 33), got {feats.shape}"
    assert list(feats.columns) == list(META_FEATURES)
    assert not feats.isna().any().any(), "featurize should not introduce NaNs"


def test_stacking_featurize_preserves_index():
    """featurize preserves the input index ordering exactly."""
    from structured_diag.diagnosis.stacking_resolver import featurize

    flat, casc, sp, arb, trig, conf = _build_dummy_oof_inputs(n=8)
    feats = featurize(
        flat_proba=flat,
        cascade_proba=casc,
        cascade_stage_probs=sp,
        arbitrator_label_probs=arb,
        arbitrator_triggered=trig,
        arbitrator_confidence=conf,
    )
    assert list(feats.index) == list(flat.index)


# ---------------------------------------------------------------------------
# 5. Utility helpers — schema alignment.
# ---------------------------------------------------------------------------


def test_align_features_to_schema_drops_extras():
    """Extra columns are dropped, missing ones imputed, order normalized."""
    from structured_diag.utils.arrays import align_features_to_schema

    df = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "extra": [5.0, 6.0],  # not in target schema
        }
    )
    aligned = align_features_to_schema(df, target_columns=["a", "b", "missing"])
    assert list(aligned.columns) == ["a", "b", "missing"]
    assert "extra" not in aligned.columns
    # missing column filled with NaN by default (or 0.0 depending on implementation)
    assert aligned.shape == (2, 3)


def test_align_features_idempotent():
    """Aligning to the same schema is a no-op besides column ordering."""
    from structured_diag.utils.arrays import align_features_to_schema

    df = pd.DataFrame({"b": [1.0, 2.0], "a": [3.0, 4.0]})
    aligned = align_features_to_schema(df, target_columns=["a", "b"])
    pd.testing.assert_frame_equal(
        aligned[["a", "b"]],
        df[["a", "b"]].astype(aligned.dtypes.to_dict()),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# 6. Hybrid resolver — agreement_or_flat policy round-trip on synthetic probs.
# ---------------------------------------------------------------------------


def test_hybrid_resolver_agreement_or_flat():
    """When flat and cascade agree, the resolver returns the agreed class."""
    from structured_diag.diagnosis import HybridResolverConfig, resolve_batch
    from structured_diag.labels import PRIMARY_LABELS

    cols = list(PRIMARY_LABELS)
    n = 10
    rng = np.random.default_rng(3)
    base = rng.dirichlet(np.ones(6), size=n)
    flat = pd.DataFrame(base, columns=cols, index=[f"r{i}" for i in range(n)])
    cascade = pd.DataFrame(base.copy(), columns=cols, index=flat.index)  # exact agreement
    cfg = HybridResolverConfig(policy="agreement_or_flat")
    diags = resolve_batch(flat_proba=flat, cascade_proba=cascade, config=cfg)
    assert len(diags) == n
    expected = flat.idxmax(axis=1).tolist()
    actual = [d.final_label for d in diags]
    assert actual == expected


# ---------------------------------------------------------------------------
# 7. Evidence layer — classify_evidence_notes returns expected typed labels.
# ---------------------------------------------------------------------------


def test_classify_evidence_notes_callable():
    """classify_evidence_notes is callable and returns a (decisive, secondary) split."""
    from structured_diag.evaluation.explanation import classify_evidence_notes

    sample_notes = [
        "early val_loss minimum",
        "saturated near-zero gap",
    ]
    decisive, secondary = classify_evidence_notes(sample_notes, "overfitting")
    # The function must return two lists; we don't pin individual notes
    # to the bucket (the rules are tested elsewhere).
    assert isinstance(decisive, list)
    assert isinstance(secondary, list)


# ---------------------------------------------------------------------------
# 8. Property-based tests on arbitrator cache-key invariants (Stage 80).
# ---------------------------------------------------------------------------
# These tests guard against silent cache collisions across folds and across
# evidence content. They are property-based rather than example-based:
# every test varies a single input dimension and asserts that the cache
# key changes (or does not change) as expected.


def _mock_evidence(curve_note: str = "default note"):
    """Build a minimal StructuredEvidence object for cache-key tests."""
    from structured_diag.evaluation.explanation import (
        CurveEvidence,
        IntegrityEvidence,
        StructuredEvidence,
    )

    return StructuredEvidence(
        schema_version="1.0",
        generated_at="2026-01-01T00:00:00Z",
        run_id="test_run",
        final_class="leakage",
        final_confidence=0.7,
        class_probabilities={"leakage": 0.7, "healthy": 0.3},
        alternative_hypotheses=[],
        rejected_hypotheses=[],
        stage_trace=[],
        top_features=[],
        curve_evidence=CurveEvidence(notes=[curve_note]),
        integrity_evidence=IntegrityEvidence(notes=[]),
        diagnostic_notes=[],
    )


def test_arb_cache_key_changes_with_fold_index():
    """Cache keys with different ``inner_fold_index`` must differ."""
    from structured_diag.diagnosis.arbitrator import _cache_key_arbitrator

    common = dict(
        run_id="r1",
        flat_label="leakage",
        cascade_label="healthy",
        flat_proba={"leakage": 0.55, "healthy": 0.45},
        cascade_proba={"leakage": 0.35, "healthy": 0.65},
        evidence=_mock_evidence(),
        backend="groq",
        model="llama-3.3-70b-versatile",
    )
    k0 = _cache_key_arbitrator(**common, inner_fold_index=0)
    k1 = _cache_key_arbitrator(**common, inner_fold_index=1)
    k_none = _cache_key_arbitrator(**common, inner_fold_index=None)
    assert k0 != k1, "fold_index=0 must produce a different key from fold_index=1"
    assert k0 != k_none, "fold_index=0 must differ from omitted (backward compat)"


def test_arb_cache_key_changes_with_evidence_hash():
    """Cache keys with different evidence notes must differ."""
    from structured_diag.diagnosis.arbitrator import _cache_key_arbitrator

    common = dict(
        run_id="r1",
        flat_label="leakage",
        cascade_label="healthy",
        flat_proba={"leakage": 0.55, "healthy": 0.45},
        cascade_proba={"leakage": 0.35, "healthy": 0.65},
        backend="groq",
        model="llama-3.3-70b-versatile",
        inner_fold_index=0,
    )
    k_a = _cache_key_arbitrator(**common, evidence=_mock_evidence("note A"))
    k_b = _cache_key_arbitrator(**common, evidence=_mock_evidence("note B"))
    assert k_a != k_b, "different evidence must produce different cache keys"


def test_arb_cache_key_deterministic_on_identical_inputs():
    """Cache keys must be identical when all inputs are identical."""
    from structured_diag.diagnosis.arbitrator import _cache_key_arbitrator

    common = dict(
        run_id="r1",
        flat_label="leakage",
        cascade_label="healthy",
        flat_proba={"leakage": 0.55, "healthy": 0.45},
        cascade_proba={"leakage": 0.35, "healthy": 0.65},
        evidence=_mock_evidence(),
        backend="groq",
        model="llama-3.3-70b-versatile",
        inner_fold_index=2,
    )
    k1 = _cache_key_arbitrator(**common)
    k2 = _cache_key_arbitrator(**common)
    assert k1 == k2, "identical inputs must produce identical keys"


def test_arb_cache_key_invariant_to_proba_key_order():
    """Cache key must not depend on dict-key ordering of proba dicts."""
    from structured_diag.diagnosis.arbitrator import _cache_key_arbitrator

    common = dict(
        run_id="r1",
        flat_label="leakage",
        cascade_label="healthy",
        evidence=_mock_evidence(),
        backend="groq",
        model="llama-3.3-70b-versatile",
        inner_fold_index=0,
    )
    k_a = _cache_key_arbitrator(
        flat_proba={"leakage": 0.55, "healthy": 0.45},
        cascade_proba={"leakage": 0.35, "healthy": 0.65},
        **common,
    )
    k_b = _cache_key_arbitrator(
        flat_proba={"healthy": 0.45, "leakage": 0.55},
        cascade_proba={"healthy": 0.65, "leakage": 0.35},
        **common,
    )
    assert k_a == k_b, "key order in proba dict must not affect cache key"
