"""Build portable exports independently of the Streamlit download controls."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform

import numpy as np
import streamlit as st

from ai_recommendation import clean_json
from cirruspp_core import aggregate_profile, profile_change_table, profile_table, quality_breakdown, quality_index


def json_bytes(value):
    return json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def export_record(value):
    """Remove supplementary audit metadata; preserve captured requests verbatim."""
    if isinstance(value, dict):
        return {key: item if key == "request" else export_record(item) for key, item in value.items()
                if key not in {"app_version", "export_schema_version", "description", "parameter_status",
                               "reference_validation", "prompt_version", "preprocessing_settings", "interpretation"}
                and not any(token in key.lower() for token in ("hash", "sha256", "fingerprint", "context_key"))}
    if isinstance(value, (list, tuple)):
        return [export_record(item) for item in value]
    return value


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")


def ai_record(result, *, selected, executed):
    """Export the captured request, never recreate an old call with today's prompt."""
    fields = ("model", "resolved_model", "input_mode", "created_at",
              "suggestion", "usage", "estimated_cost_usd", "response_id")
    record = {key: result[key] for key in fields if key in result}
    record.update(selected=selected, executed=executed,
                  goal=result.get("context", {}).get("user_goal"), request=result.get("request"))
    if record["request"] is None:
        record["request_status"] = "not_recorded_in_legacy_session"
        record["legacy_context"] = result.get("context")
    return record


def ai_identity(result):
    return (result.get("context_key"), result.get("model"), result.get("created_at"))


def build_exports(*, raw, canonical, run, pipeline_key, spec, order_key, stage_order,
                  selected_source, context, weights, recommendations,
                  comparison=None, ai_suggestions=None, selected_ai_result=None):
    valid = run is not None and run.get("key") == pipeline_key
    before = aggregate_profile(canonical, context["sampling_rate_hz"])
    after = aggregate_profile(run["cleaned"], context["sampling_rate_hz"]) if valid else None
    selected = spec.to_dict()
    executed = run["meta"]["spec"] if valid else None
    audit = {
        "tool": "CIRRUS++",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), **{p: version(p) for p in
                     ("streamlit", "pandas", "numpy", "scipy", "scikit-learn", "plotly")}},
        **context, "time_axis": canonical.attrs.get("time_axis", context.get("time_axis", {})),
        "time_processing": canonical.attrs.get("time_processing", {}), "execution_status": "executed" if valid else "not_run", "rows": len(raw),
        "original_columns": list(raw.columns), "quality_weights": weights,
        "quality_index_semantics": "domain-weighted intrinsic diagnostic heuristic; not ground truth",
        "quality_index_before": quality_index(before, weights),
        "quality_index_after_cleaning": quality_index(after, weights) if valid else None,
        "quality_breakdown_before": quality_breakdown(before, weights).to_dict("records"),
        "quality_breakdown_after_cleaning": quality_breakdown(after, weights).to_dict("records") if valid else None,
        "profile_before": before, "profile_after_cleaning": after,
        "heuristic_recommendations": {stage: {key: value for key, value in rec.as_dict().items()
            if key in {"method", "rule_trace"}} for stage, rec in recommendations.items()},
        "selected_pipeline": selected, "selected_pipeline_source": selected_source,
        "selected_order_key": order_key,
        "executed_pipeline": executed,
        "executed_order_key": run["meta"]["order"] if valid else None,
        "executed_pipeline_source": run.get("source", "Custom") if valid else None,
        "execution_meta": {k: run["meta"][k] for k in
            ("seed", "method_parameters", "scaler_parameters", "stage_changed_fraction",
             "cleaning_changed_fraction", "scaling_changed_fraction", "overall_changed_fraction")
            if k in run["meta"]} if valid else None,
        "executed_at_utc": run.get("executed_at_utc") if valid else None,
        "execution_seconds": run.get("elapsed") if valid else None,

        "csv_format": {"encoding": "UTF-8", "separator": ",", "decimal": ".", "missing_values": "empty field"},
    }
    files = {}
    if valid:
        full = raw.copy()
        for col in context["signals"]:
            full[col] = run["processed"][col].to_numpy()
        files["cirruspp_processed_full.csv"] = csv_bytes(full)
        metrics = profile_change_table(canonical, run["cleaned"], context["sampling_rate_hz"], weights)
        audit["feature_metrics"] = metrics.to_dict("records")
        audit["remaining_missing_values"] = int((~np.isfinite(run["processed"][context["signals"]].to_numpy(float))).sum())
    else:
        audit["feature_metrics"] = [{"feature": row["feature"],
            **{f"before_{key}": value for key, value in row.items() if key != "feature"}}
            for row in profile_table(canonical, context["sampling_rate_hz"], weights).to_dict("records")]
    if comparison:
        comparison_context = comparison.get("context") or {
            "context_status": "unavailable_in_legacy_session",
            "note": "Saved before comparison context was recorded. Rerun the comparison to include its settings.",
        }
        audit["method_comparison"] = {**comparison_context, "results": comparison["table"].to_dict("records")}
    # Current suggestions and the executed suggestion are recorded once, with
    # independent flags: a generated recommendation is not an executed pipeline.
    executed_ai = run.get("ai_recommendation") if valid else None
    candidates = list(ai_suggestions or [])
    if executed_ai is not None:
        candidates.append(executed_ai)
    if selected_ai_result is not None:
        candidates.append(selected_ai_result)
    unique = {ai_identity(result): result for result in candidates}
    audit["ai_recommendations"] = [ai_record(result,
        selected=selected_ai_result is not None and identity == ai_identity(selected_ai_result),
        executed=executed_ai is not None and identity == ai_identity(executed_ai))
        for identity, result in unique.items()]
    # Removed controls and internal experiment plumbing do not describe the run.
    for key in ("eye_tracking_validity_mode", "eye_tracking_validity_meta", "domain_label",
                "timestamp_column", "timestamp_unit_selection"):
        audit.pop(key, None)
    audit = export_record(audit)
    files["cirruspp_audit.json"] = json_bytes(audit)
    return {"valid_run": valid, "audit": clean_json(audit), "files": files}


def render_export(bundle):
    files = bundle["files"]
    st.subheader("Export")
    st.write("Download your processed data and the decisions and statistics behind it.")
    if not bundle["valid_run"]:
        st.info("Run the current pipeline in 5 · Pipeline lab to download processed data. The JSON already contains your current settings and input statistics, marked as not run.")
    else:
        st.caption("The exports match the current data and executed pipeline.")
    data, record = st.columns(2, gap="large")
    with data.container(border=True):
        st.subheader("Processed CSV")
        st.write("All rows and columns, with selected signals replaced by the processed output. Original timestamps are retained; selected normalization is included.")
        st.download_button("Download processed CSV", files.get("cirruspp_processed_full.csv", b""),
                           "cirruspp_processed_full.csv", "text/csv", disabled=not bundle["valid_run"],
                           key="export_processed_csv", width="stretch")
    with record.container(border=True):
        st.subheader("Decisions & statistics JSON")
        st.write("Recommendations, current selection and actual execution; captured AI requests, method parameters, complete per-signal statistics and current method-comparison results.")
        st.download_button("Download decisions & statistics JSON", files["cirruspp_audit.json"],
                           "cirruspp_audit.json", "application/json", key="export_audit_json", width="stretch")
    st.caption("The quality index is a diagnostic heuristic, not proof of reconstruction accuracy. Only current method-comparison results are included in the JSON.")
    with st.expander("Preview JSON"):
        st.json(bundle["audit"], expanded=False)
