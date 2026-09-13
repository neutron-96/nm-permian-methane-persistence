
# ============================================================================
# FULL CHECKPOINT SCRIPT — refit and save every model, config, and dataset
# Run this ONCE now. After this, the zip is fully self-contained: you can
# load any model/config/dataset directly without rerunning anything upstream.
# ============================================================================
import os, json, shutil, glob
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from lifelines import CoxPHFitter

WORK = "/kaggle/working"
CKPT = f"{WORK}/checkpoint"
for sub in ["models", "configs", "code"]:
    os.makedirs(f"{CKPT}/{sub}", exist_ok=True)

print("="*70, "\nSTEP 1 — Persistence classifier (matches paper Table I exactly)\n", "="*70)

df_k5 = pd.read_parquet(f"{WORK}/nm_features_k5_v2.parquet")
df_k5["target_persistent"] = df_k5["persistence_class"].isin(["persistent", "ultra_persistent"]).astype(int)

PERSIST_FEATURE_COLS = ["n_obs_early", "n_detect_early", "detect_rate_early",
    "span_days_early", "first_gap_days", "first_emission_early",
    "max_emission_early", "mean_emission_early", "n_emission_readings_early"]
X = df_k5[PERSIST_FEATURE_COLS].copy()
for col in ["first_emission_early", "max_emission_early", "mean_emission_early", "first_gap_days"]:
    X[f"{col}_missing"] = X[col].isna().astype(int)
    X[col] = X[col].fillna(X[col].median())
y = df_k5["target_persistent"].values

PERSIST_CONFIG = {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 200,
    "reg_lambda": 5.0, "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8}
pw_full = (y == 0).sum() / (y == 1).sum()

persist_model = CalibratedClassifierCV(
    xgb.XGBClassifier(**PERSIST_CONFIG, scale_pos_weight=pw_full, tree_method="hist",
                       eval_metric="aucpr", random_state=42),
    method="isotonic", cv=3,
)
persist_model.fit(X, y)
joblib.dump(persist_model, f"{CKPT}/models/persistence_classifier_calibrated.pkl")
joblib.dump(list(X.columns), f"{CKPT}/models/persistence_classifier_feature_order.pkl")
print(f"Saved persistence classifier. Feature order: {list(X.columns)}")

with open(f"{CKPT}/configs/persistence_classifier_config.json", "w") as f:
    json.dump({"config": PERSIST_CONFIG, "feature_cols": PERSIST_FEATURE_COLS,
               "window_k": 5, "positive_class_rate": float(y.mean()),
               "reported_metrics": {"AUROC": "0.772±0.042", "AUPRC": "0.564±0.055",
                                     "Brier": "0.080±0.006",
                                     "nested_CV_AUROC": "0.761±0.027", "nested_CV_AUPRC": "0.577±0.044"}}, f, indent=2)

print("\n" + "="*70, "\nSTEP 2 — MAG-risk classifiers (two binary targets)\n", "="*70)

df_mag = pd.read_parquet(f"{WORK}/nm_sources_final_mag.parquet")
df_merged = df_k5.merge(
    df_mag[["source_name", "mag_class", "distance_km"]].rename(columns={"source_name": "source_id"}),
    on="source_id", how="inner")

MAG_FEATURE_COLS = ["n_detect_early", "detect_rate_early", "span_days_early", "first_gap_days"]
X_mag = df_merged[MAG_FEATURE_COLS].copy()
X_mag["first_gap_days"] = X_mag["first_gap_days"].fillna(X_mag["first_gap_days"].median())

MAG_CONFIG = {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 200,
    "reg_lambda": 5.0, "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8}

for target_name, target_class in [("directly_comparable", "directly_comparable"),
                                    ("no_public_match", "no_public_match")]:
    y_mag = (df_merged["mag_class"] == target_class).astype(int).values
    pw = (y_mag == 0).sum() / (y_mag == 1).sum()
    m = xgb.XGBClassifier(**MAG_CONFIG, scale_pos_weight=pw, tree_method="hist",
                           eval_metric="aucpr", random_state=42)
    m.fit(X_mag, y_mag)
    joblib.dump(m, f"{CKPT}/models/mag_classifier_{target_name}.pkl")
    print(f"Saved MAG classifier for target: {target_name} (positive rate {y_mag.mean():.3f})")

joblib.dump(MAG_FEATURE_COLS, f"{CKPT}/models/mag_classifier_feature_order.pkl")
with open(f"{CKPT}/configs/mag_classifier_config.json", "w") as f:
    json.dump({"config": MAG_CONFIG, "feature_cols": MAG_FEATURE_COLS,
               "reported_metrics": {
                   "directly_comparable": {"AUROC": "0.822±0.039", "AUPRC": "0.420±0.088"},
                   "no_public_match": {"AUROC": "0.798±0.020", "AUPRC": "0.500±0.043"}}}, f, indent=2)

print("\n" + "="*70, "\nSTEP 3 — Cox survival model (final, instrument-adjusted, PH-corrected)\n", "="*70)

df_surv = pd.read_parquet(f"{WORK}/nm_survival_data.parquet")
df_cox = df_surv[df_surv["duration_days"] > 0.01].copy()

df_sources_lbl = pd.read_parquet(f"{WORK}/nm_sources_labeled.parquet")
df_plumes = pd.read_parquet(f"{WORK}/nm_plumes_clean.parquet")
KNOWN_MISMATCH_IDS = ["CH4_1B2_250m_-104.11754_32.05183?status=not_deleted",
                       "CH4_1B2_250m_-104.11947_32.02271?status=not_deleted"]
df_sources_clean = df_sources_lbl[~df_sources_lbl["source_name"].isin(KNOWN_MISMATCH_IDS)].copy()
plume_dt_lookup = dict(zip(df_plumes["plume_id"], df_plumes["datetime"]))
plume_instr_lookup = dict(zip(df_plumes["plume_id"], df_plumes["instrument"]))

first_instr = []
for _, row in df_sources_clean.iterrows():
    pids = list(row["plume_ids"]) if isinstance(row["plume_ids"], (list, np.ndarray)) else []
    pid_dates = [(pid, plume_dt_lookup.get(pid)) for pid in pids if pid in plume_dt_lookup]
    if not pid_dates:
        continue
    pid_dates.sort(key=lambda x: x[1])
    first_instr.append({"source_id": row["source_name"], "first_instrument": plume_instr_lookup.get(pid_dates[0][0])})
df_first_instr = pd.DataFrame(first_instr)

df_cox_instr = df_cox.merge(df_first_instr, on="source_id", how="left")
df_cox_instr["instrument_grouped"] = df_cox_instr["first_instrument"].apply(
    lambda x: x if x in ["GAO", "ang", "av3"] else "other_small")

COX_COVARIATES = ["detect_rate_early", "span_days_early", "first_gap_days"]
instr_dummies = pd.get_dummies(df_cox_instr["instrument_grouped"], prefix="instr", drop_first=True)
model_df = pd.concat([df_cox_instr[["duration_days", "event_observed"] + COX_COVARIATES], instr_dummies], axis=1).dropna()
model_df["instr_av3"] = model_df["instr_av3"].astype(int)
COX_REMAINING_COVARIATES = COX_COVARIATES + [c for c in instr_dummies.columns if c != "instr_av3"]

cox_model = CoxPHFitter()
cox_model.fit(model_df[["duration_days", "event_observed"] + COX_REMAINING_COVARIATES + ["instr_av3"]],
              duration_col="duration_days", event_col="event_observed", strata=["instr_av3"])
joblib.dump(cox_model, f"{CKPT}/models/cox_survival_model.pkl")
model_df.to_parquet(f"{CKPT}/models/cox_model_training_data.parquet")
print("Saved Cox survival model and its exact training data.")

with open(f"{CKPT}/configs/cox_model_config.json", "w") as f:
    json.dump({"covariates": COX_REMAINING_COVARIATES, "strata": ["instr_av3"],
               "n_sources": len(model_df),
               "reported_coefficients": cox_model.summary[["coef", "exp(coef)", "p"]].round(4).to_dict()}, f, indent=2, default=str)

print("\n" + "="*70, "\nSTEP 4 — Mitigation scoring config (deterministic, no model object needed)\n", "="*70)

with open(f"{CKPT}/configs/mitigation_scoring_weights.json", "w") as f:
    json.dump({
        "S1_emissions_only": {"emission_rank": 1.0},
        "S2_persistence_aware": {"emission_rank": 0.6, "persistence_rank_pct": 0.4},
        "S3_accountability_aware": {"emission_rank": 0.5, "persistence_rank_pct": 0.3, "mag_rank_pct": 0.2},
        "S4_justice_aware": {"emission_rank": 0.4, "persistence_rank_pct": 0.25, "mag_rank_pct": 0.15, "vulnerability_rank_pct": 0.2},
        "note": "All components are percentile ranks (0-1) of the underlying variable BEFORE weighting, not raw values.",
    }, f, indent=2)
print("Saved mitigation scoring weights config.")

print("\n" + "="*70, "\nSTEP 5 — Master threshold/design-decision log (every judgment call made)\n", "="*70)

with open(f"{CKPT}/configs/analysis_decisions_log.json", "w") as f:
    json.dump({
        "persistence_labeling": {"min_reliable_observations": 5, "min_ultra_persistent_observations": 10,
                                    "intermittent_threshold": 0.25, "persistent_threshold": 0.5,
                                    "sensitivity_low": {"min_reliable": 2, "min_ultra": 5},
                                    "sensitivity_high": {"min_reliable": 10, "min_ultra": 15}},
        "mag_taxonomy": {"high_confidence_km": 1.0, "medium_confidence_km": 5.0, "low_confidence_km": 15.0,
                           "site_resolved_segments": ["W-PROC", "W-NGTC", "W-LDC", "W-UNSTG"],
                           "basin_aggregated_segments": ["W-ONSH", "W-GB"]},
        "study_region_bbox": {"lon_min": -104.6, "lon_max": -103.0, "lat_min": 32.0, "lat_max": 32.8,
                                 "note": "lat_min raised from 31.4 to 32.0 to exclude Texas after border check"},
        "ch4_gwp_used": 25,
        "exposure_buffer_radii_km": [2, 5, 10],
        "ghgrp_year_range": "2011-2023 (2024/2025 confirmed absent nationwide, not NM-specific)",
    }, f, indent=2)
print("Saved full decision log.")

print("\n" + "="*70, "\nDONE. All models, configs, and decisions checkpointed.\n", "="*70)
