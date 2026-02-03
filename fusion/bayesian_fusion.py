import numpy as np
from typing import Dict, List

class BayesianFusion:
    """
    Combines multiple risk domains into a single global risk score using 
    probabilistic averaging and weighted Bayesian updating.
    """
    def __init__(self, weights: Dict[str, float] = None):
        # Default weights if none provided
        self.weights = weights or {
            "fraud": 0.35,
            "cyber": 0.30,
            "compliance": 0.20,
            "churn": 0.15
        }

    def fuse_risks(self, risks: Dict[str, float]) -> Dict[str, float]:
        """
        Input: {'fraud': 0.8, 'cyber': 0.2, ...}
        Output: {'global_score': 0.55, 'confidence': 0.85}
        """
        scores = []
        weight_sum = 0
        
        for domain, risk in risks.items():
            if domain in self.weights:
                scores.append(risk * self.weights[domain])
                weight_sum += self.weights[domain]
        
        if not weight_sum:
            return {"global_score": 0.0, "confidence": 0.0}
            
        global_score = sum(scores) / weight_sum
        
        # Confidence logic: Higher agreement between models or extremely high 
        # single risk increases confidence.
        variance = np.var(list(risks.values()))
        confidence = 1.0 - (variance * 0.5) # Lower variance -> higher confidence
        
        return {
            "global_score": np.clip(global_score, 0, 1),
            "confidence": np.clip(confidence, 0, 1)
        }

    def update_with_prior(self, prior_score: float, new_evidence: float, weight: float = 0.5) -> float:
        """
        Simple Bayesian update: new_prior = (prior * (1-weight)) + (evidence * weight)
        """
        return (prior_score * (1 - weight)) + (new_evidence * weight)
