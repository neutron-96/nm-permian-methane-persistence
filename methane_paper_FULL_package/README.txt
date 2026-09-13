
HOW TO USE THIS PACKAGE WITHOUT RERUNNING THE NOTEBOOK
========================================================

/models/  -- pickled, ready-to-use model objects:
  persistence_classifier_calibrated.pkl  -> joblib.load(), then .predict_proba(X)
  persistence_classifier_feature_order.pkl -> exact column order/names X must have
  mag_classifier_directly_comparable.pkl
  mag_classifier_no_public_match.pkl
  mag_classifier_feature_order.pkl
  cox_survival_model.pkl                 -> lifelines CoxPHFitter, use .predict_partial_hazard()
  cox_model_training_data.parquet        -> exact data the Cox model was fit on

/configs/ -- every hyperparameter, threshold, and design decision as plain JSON:
  persistence_classifier_config.json
  mag_classifier_config.json
  cox_model_config.json
  mitigation_scoring_weights.json
  analysis_decisions_log.json            -> every threshold choice made during the analysis

/code/checkpoint_and_refit_script.py     -> the exact code that produced everything above

/study1_*, /study2_*  -- every intermediate and final dataset (parquet/csv)

To troubleshoot or revise for a journal response WITHOUT rerunning the pipeline:
  import joblib
  model = joblib.load("models/persistence_classifier_calibrated.pkl")
  cols = joblib.load("models/persistence_classifier_feature_order.pkl")
  # apply model to any new/modified dataframe with those exact columns
