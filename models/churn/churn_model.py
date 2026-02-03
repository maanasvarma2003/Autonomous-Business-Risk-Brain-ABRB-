import xgboost as xgb
import joblib
import os
import pandas as pd
import numpy as np

class ChurnModel:
    """
    Churn Prediction Model: Gradient Boosting for customer churn.
    """
    def __init__(self, model_dir: str = "models/churn"):
        self.model_dir = model_dir
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        print(f"Training Churn Model on {len(X)} samples...")
        
        pos_count = sum(y == 1)
        neg_count = sum(y == 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.save_model()
        print("Churn Model training complete.")

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save_model(self):
        joblib.dump(self.model, os.path.join(self.model_dir, "churn_xgb.joblib"))

    def load_model(self):
        self.model = joblib.load(os.path.join(self.model_dir, "churn_xgb.joblib"))
