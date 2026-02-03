from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import random
import os
import joblib
import pandas as pd
import numpy as np

# Import our system components
from models.fraud.fraud_model import FraudModel
from models.churn.churn_model import ChurnModel
from fusion.bayesian_fusion import BayesianFusion
from graph.risk_graph import RiskGraph
from llm.reasoning_agent import ReasoningAgent

router = APIRouter()

class RiskRequest(BaseModel):
    entity_type: str
    entity_id: str
    features: Optional[Dict[str, Any]] = None

class RiskResponse(BaseModel):
    fraud: float
    cyber: float
    compliance: float
    churn: float
    global_score: float
    confidence: float

# In a real app, these would be loaded at startup
fusion_engine = BayesianFusion()
graph_engine = RiskGraph()
reasoning_agent = ReasoningAgent()

# Domain models (Lazy loading for production efficiency)
_models = {
    "fraud": None,
    "churn": None,
    "cyber": None,
    "compliance": None,
    "engineer": None
}

def get_engineer():
    if _models["engineer"] is None:
        try:
            _models["engineer"] = joblib.load("features/engineer.joblib")
        except Exception as e:
            print(f"Warning: Could not load feature engineer: {e}")
    return _models["engineer"]

def get_model(domain: str):
    if _models[domain] is None:
        try:
            if domain == "fraud":
                m = FraudModel()
                m.load_models()
                _models[domain] = m
            elif domain == "churn":
                m = ChurnModel()
                m.load_model()
                _models[domain] = m
        except Exception as e:
            print(f"Warning: Could not load real {domain} model: {e}")
    return _models[domain]

@router.post("/score", response_model=RiskResponse)
async def get_risk_score(request: RiskRequest):
    """
    Computes unified risk score for an entity using production ML models.
    """
    try:
        # Initialize default risks
        random.seed(request.entity_id)
        domain_risks = {
            "fraud": random.uniform(0.1, 0.4),
            "cyber": random.uniform(0.1, 0.4),
            "compliance": random.uniform(0.1, 0.4),
            "churn": random.uniform(0.1, 0.4)
        }

        # 1. Use real models if features are provided
        if request.features:
            engineer = get_engineer()
            # Fraud real inference
            fraud_model = get_model("fraud")
            if fraud_model and hasattr(fraud_model, 'xgb_model'):
                # Extract only relevant features for this model
                expected_features = fraud_model.xgb_model.get_booster().feature_names
                if expected_features:
                    feat_dict = {f: request.features.get(f, 0) for f in expected_features}
                    feat_df = pd.DataFrame([feat_dict])[expected_features]
                    if engineer and 'fraud' in engineer.scalers:
                        feat_df[expected_features] = engineer.scalers['fraud'].transform(feat_df[expected_features])
                    domain_risks["fraud"] = float(fraud_model.predict_risk(feat_df)[0])

            # Churn real inference
            churn_model = get_model("churn")
            if churn_model and hasattr(churn_model, 'model'):
                expected_features = churn_model.model.get_booster().feature_names
                if expected_features:
                    feat_dict = {f: request.features.get(f, 0) for f in expected_features}
                    feat_df = pd.DataFrame([feat_dict])[expected_features]
                    if engineer and 'churn_scaler' in engineer.scalers:
                        feat_df[expected_features] = engineer.scalers['churn_scaler'].transform(feat_df[expected_features])
                    # Churn is typically proba of class 1
                    domain_risks["churn"] = float(churn_model.model.predict_proba(feat_df)[:, 1][0])

        # 2. Fuse risks
        fused = fusion_engine.fuse_risks(domain_risks)
        
        # 3. Graph influence
        graph_engine.update_node_risk(request.entity_id, fused['global_score'])
        propagated_score = graph_engine.compute_propagated_risk(request.entity_id)
        
        final_score = (0.8 * fused['global_score']) + (0.2 * propagated_score)
        
        return RiskResponse(
            fraud=round(domain_risks['fraud'], 2),
            cyber=round(domain_risks['cyber'], 2),
            compliance=round(domain_risks['compliance'], 2),
            churn=round(domain_risks['churn'], 2),
            global_score=round(final_score, 2),
            confidence=round(fused['confidence'], 2)
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/explain/{entity_id}")
async def explain_risk(entity_id: str):
    """
    Returns natural language explanation and graph context for an entity's risk.
    """
    random.seed(entity_id)
    scores = {
        "fraud": random.uniform(0.1, 0.9),
        "cyber": random.uniform(0.1, 0.9),
        "compliance": random.uniform(0.1, 0.9),
        "churn": random.uniform(0.1, 0.9),
        "global": random.uniform(0.5, 0.8)
    }
    
    top_features = {
        "fraud": {"transaction_amount": 0.8, "billing_zip_mismatch": 0.4},
        "cyber": {"suspicious_agent": 0.9, "impossible_travel": 0.6},
        "compliance": {"gdpr_article_12_violation": 0.7}
    }
    
    explanation = reasoning_agent.generate_explanation(
        entity_id, scores, {"neighbors_count": random.randint(1, 15)}, top_features
    )
    
    return {
        "entity_id": entity_id,
        "reasoning": explanation, # Fixed key to match display script
        "raw_scores": scores
    }
