import xgboost as xgb
from sklearn.ensemble import IsolationForest
import joblib
import os
import pandas as pd
import numpy as np
from typing import Dict

class FraudModel:
    """
    Hybrid Fraud Detection Model:
    - Isolation Forest for unsupervised anomaly detection
    - XGBoost for supervised classification
    """
    def __init__(self, model_dir: str = "models/fraud"):
        self.model_dir = model_dir
        self.xgb_model = None
        self.iso_forest = None
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        print(f"Training Fraud Model on {len(X)} samples with {X.shape[1]} features...")
        print("Training Isolation Forest for anomaly detection...")
        self.iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        self.iso_forest.fit(X)
        
        print("Training XGBoost with imbalance handling...")
        # Calculate scale_pos_weight for better precision/recall on fraud
        pos_count = sum(y == 1)
        neg_count = sum(y == 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=500, # Increased for better accuracy
            max_depth=10,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, # Handles imbalance
            random_state=42,
            eval_metric='aucpr', # Optimize for precision-recall area
            n_jobs=-1
        )
        self.xgb_model.fit(X, y)
        self.save_models()
        print("Fraud Model training complete.")

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Combines anomaly score and classification probability.
        Returns risk ∈ [0,1]
        """
        # Isolation forest returns -1 for anomalies, 1 for normal
        # We transform it to [0,1] where 1 is anomalous
        ano_scores = self.iso_forest.decision_function(X)
        ano_risk = 1 - (ano_scores - ano_scores.min()) / (ano_scores.max() - ano_scores.min() + 1e-6)
        
        # XGBoost probability
        prob_fraud = self.xgb_model.predict_proba(X)[:, 1]
        
        # Combined risk (weighted average)
        combined_risk = 0.4 * ano_risk + 0.6 * prob_fraud
        return np.clip(combined_risk, 0, 1)

    def save_models(self):
        joblib.dump(self.xgb_model, os.path.join(self.model_dir, "xgb_fraud.joblib"))
        joblib.dump(self.iso_forest, os.path.join(self.model_dir, "iso_forest.joblib"))

    def load_models(self):
        self.xgb_model = joblib.load(os.path.join(self.model_dir, "xgb_fraud.joblib"))
        self.iso_forest = joblib.load(os.path.join(self.model_dir, "iso_forest.joblib"))
