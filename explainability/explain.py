import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class RiskExplainer:
    """
    Generates SHAP explanations for ML model predictions.
    """
    def __init__(self):
        self.explainers = {}

    def explain_fraud(self, model, X: pd.DataFrame) -> Dict[str, float]:
        """
        Explains XGBoost fraud model.
        """
        if 'fraud' not in self.explainers:
            self.explainers['fraud'] = shap.TreeExplainer(model)
        
        shap_values = self.explainers['fraud'].shap_values(X)
        
        # Return top 5 contributing features
        feature_importance = dict(zip(X.columns, shap_values[0]))
        sorted_features = dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
        
        return sorted_features

    def explain_churn(self, model, X: pd.DataFrame) -> Dict[str, float]:
        """
        Explains Churn model.
        """
        if 'churn' not in self.explainers:
            self.explainers['churn'] = shap.TreeExplainer(model)
            
        shap_values = self.explainers['churn'].shap_values(X)
        feature_importance = dict(zip(X.columns, shap_values[0]))
        return dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)[:5])

    def get_summary_explanation(self, domain_explanations: Dict[str, Dict[str, float]]) -> str:
        """
        Aggregates multiple domain explanations into a text summary.
        """
        summary = "Primary Risk Drivers:\n"
        for domain, features in domain_explanations.items():
            top_feat = list(features.keys())[0] if features else "N/A"
            summary += f"- {domain.capitalize()}: Top driver is '{top_feat}'\n"
        return summary
