from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier


FEATURE_NAMES = ["sequence_length", "mean_correct", "recent_mean_correct", "unique_skills"]


def aggregate_sequence_features(sequences: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    for seq in sequences:
        corrects = np.asarray(seq.get("corrects", []), dtype=np.float32)
        skills = seq.get("skills", [])
        if len(corrects) < 4:
            continue
        rows.append(
            [
                len(corrects),
                float(corrects[:-1].mean()),
                float(corrects[max(0, len(corrects) - 6) : -1].mean()),
                len(set(skills[:-1])),
            ]
        )
        targets.append(int(corrects[-1] > 0))
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.int32)


def save_optional_shap_summary(sequences: list[dict[str, Any]], output_path: str | Path) -> dict[str, Any]:
    """Optional SHAP hook over simple aggregated features.

    This intentionally avoids forcing SHAP into the AKT sequence model training
    path. If SHAP is unavailable, a feature-importance fallback is saved.
    """

    X, y = aggregate_sequence_features(sequences)
    if len(X) < 20 or len(set(y.tolist())) < 2:
        summary = {"status": "skipped", "reason": "Not enough varied sequence examples for SHAP proxy."}
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        summary = {
            "status": "feature_importance_fallback",
            "feature_names": FEATURE_NAMES,
            "feature_importances": model.feature_importances_.tolist(),
        }
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[: min(200, len(X))])
            values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
            summary = {
                "status": "ok",
                "feature_names": FEATURE_NAMES,
                "mean_abs_shap": np.abs(values).mean(axis=0).tolist(),
            }
        except Exception as exc:
            summary["shap_error"] = str(exc)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

