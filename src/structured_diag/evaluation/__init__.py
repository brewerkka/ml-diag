# ruff: noqa: I001
# Import order matters here: ``metrics`` exports ``ClassificationReport``
# and ``classification_report``, which ``models.flat_baseline`` imports
# at module load. ``compare_flat_vs_hier`` (and ``case_outputs``,
# ``explanation``) in turn import ``models.flat_baseline``, so they must
# load AFTER ``metrics`` is fully exported by this package. Putting the
# imports in alphabetical order breaks the cycle — do not auto-sort.
from structured_diag.evaluation.metrics import (
    ClassificationReport,
    bootstrap_delta_ci,
    bootstrap_delta_ci_grouped,
    bootstrap_metric_ci,
    brier_score_multiclass,
    classification_report,
    expected_calibration_error,
    holm_bonferroni_adjust,
    maximum_calibration_error,
    reliability_diagram_bins,
)
from structured_diag.evaluation.reports import (
    render_flat_baseline_markdown,
    report_to_markdown,
    reports_to_json,
    write_report,
)
from structured_diag.evaluation.compare_flat_vs_hier import (
    SliceComparison,
    compare_all_slices,
    compare_on_slice,
    render_comparison_markdown,
)
from structured_diag.evaluation.explanation import (
    EVIDENCE_SCHEMA_VERSION,
    CurveEvidence,
    FeatureContribution,
    IntegrityEvidence,
    StageTraceEntry,
    StructuredEvidence,
    build_evidence,
    classify_evidence_notes,
    render_json,
    render_markdown,
    write_evidence,
)
from structured_diag.evaluation.error_attribution import (
    AttributionSummary,
    DisagreementRow,
    RowAttribution,
    attribute_errors,
    find_disagreements,
    render_attribution_markdown,
    render_disagreements_markdown,
    summarize_attributions,
)
from structured_diag.evaluation.case_outputs import (
    CASE_OUTPUTS_SCHEMA_VERSION,
    REQUIRED_FILES,
    SYSTEM_NAME,
    extract_recommendations_payload,
    validate_case_dir,
    write_case_outputs,
)

__all__ = [
    "CASE_OUTPUTS_SCHEMA_VERSION",
    "ClassificationReport",
    "CurveEvidence",
    "EVIDENCE_SCHEMA_VERSION",
    "FeatureContribution",
    "IntegrityEvidence",
    "REQUIRED_FILES",
    "SYSTEM_NAME",
    "AttributionSummary",
    "DisagreementRow",
    "RowAttribution",
    "SliceComparison",
    "StageTraceEntry",
    "StructuredEvidence",
    "attribute_errors",
    "bootstrap_delta_ci",
    "bootstrap_delta_ci_grouped",
    "bootstrap_metric_ci",
    "brier_score_multiclass",
    "holm_bonferroni_adjust",
    "maximum_calibration_error",
    "reliability_diagram_bins",
    "build_evidence",
    "find_disagreements",
    "classification_report",
    "compare_all_slices",
    "compare_on_slice",
    "expected_calibration_error",
    "extract_recommendations_payload",
    "render_attribution_markdown",
    "render_comparison_markdown",
    "render_disagreements_markdown",
    "render_flat_baseline_markdown",
    "render_json",
    "render_markdown",
    "report_to_markdown",
    "summarize_attributions",
    "reports_to_json",
    "validate_case_dir",
    "write_case_outputs",
    "write_evidence",
    "write_report",
]
