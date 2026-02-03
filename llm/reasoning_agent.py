import os
import json
from typing import Dict, Any

class ReasoningAgent:
    """
    Produces natural language reasoning for the global risk score.
    Uses an OpenAI-compatible interface but defaults to a deterministic template.
    """
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.api_key = os.getenv("OPENAI_API_KEY")

    def generate_explanation(self, entity_id: str, scores: Dict[str, float], graph_context: Dict[str, Any], top_features: Dict[str, Any]) -> str:
        """
        Main entry point for generating a narrative explanation.
        """
        if self.use_llm and self.api_key:
            return self._call_llm(entity_id, scores, graph_context, top_features)
        else:
            return self._template_reasoning(entity_id, scores, graph_context, top_features)

    def _template_reasoning(self, entity_id: str, scores: Dict[str, float], graph_context: Dict[str, Any], top_features: Dict[str, Any]) -> str:
        """
        Deterministic prompt-style template.
        """
        global_risk = scores.get('global', 0)
        level = "CRITICAL" if global_risk > 0.8 else "HIGH" if global_risk > 0.6 else "MEDIUM" if global_risk > 0.3 else "LOW"
        
        explanation = f"Entity {entity_id} is currently at {level} risk ({global_risk:.2f}).\n\n"
        
        # Domain breakdown
        explanation += "Analysis Breakdown:\n"
        for domain in ['fraud', 'cyber', 'compliance', 'churn']:
            score = scores.get(domain, 0)
            if score > 0.5:
                explanation += f"- Elevated {domain} risk detected ({score:.2f}). "
                if domain in top_features:
                    top_f = list(top_features[domain].keys())[0]
                    explanation += f"Primary driver: {top_f}.\n"
        
        # Graph context
        if graph_context:
            conn_count = graph_context.get('neighbors_count', 0)
            explanation += f"\nGraph Context: This entity has {conn_count} active connections. "
            if conn_count > 10:
                explanation += "High degree of connectivity increases risk propagation potential."
                
        return explanation

    def _call_llm(self, entity_id: str, scores: Dict[str, float], graph_context: Dict[str, Any], top_features: Dict[str, Any]) -> str:
        # Placeholder for actual OpenAI API call
        # Since we want to avoid external dependencies failing in this environment, 
        # we stick to the template unless explicitly requested.
        return self._template_reasoning(entity_id, scores, graph_context, top_features)
