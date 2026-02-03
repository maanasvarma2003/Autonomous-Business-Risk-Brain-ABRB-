import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class RiskGraph:
    """
    Graph Risk Engine: Models relationships between entities and propagates risk.
    """
    def __init__(self):
        self.graph = nx.Graph()

    def add_relationship(self, entity_a: str, entity_b: str, relationship_type: str, weight: float = 1.0):
        """
        Adds a relationship between two entities (e.g., User -> IP, User -> Vendor).
        """
        self.graph.add_edge(entity_a, entity_b, type=relationship_type, weight=weight)

    def update_node_risk(self, entity_id: str, direct_risk: float):
        """
        Updates the intrinsic risk score of a node.
        """
        if entity_id in self.graph:
            self.graph.nodes[entity_id]['direct_risk'] = direct_risk
        else:
            self.graph.add_node(entity_id, direct_risk=direct_risk)

    def compute_propagated_risk(self, entity_id: str) -> float:
        """
        Computes risk based on neighbors and centrality.
        """
        if entity_id not in self.graph:
            return 0.0
            
        # 1. Degree Centrality as a base importance measure
        centrality = nx.degree_centrality(self.graph).get(entity_id, 0)
        
        # 2. PageRank-inspired risk propagation
        # We simulate a "risk walk" where risk spreads through connections
        # For simplicity in this real-time engine, we use a weighted neighbor risk
        neighbor_risks = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_weight = self.graph[entity_id][neighbor].get('weight', 1.0)
            neighbor_risk = self.graph.nodes[neighbor].get('direct_risk', 0.0)
            
            # Penalize risks coming from high-degree neighbors (risk dispersion)
            # but reward risks from highly central neighbors (risk concentration)
            neighbor_risks.append(neighbor_risk * edge_weight)
            
        avg_neighbor_risk = sum(neighbor_risks) / len(neighbor_risks) if neighbor_risks else 0.0
        
        # 3. Closeness to known high-risk nodes (simplified)
        # Total risk = 60% direct + 30% neighbor influence + 10% importance (centrality)
        direct_risk = self.graph.nodes[entity_id].get('direct_risk', 0.0)
        
        total_risk = (0.6 * direct_risk) + (0.3 * avg_neighbor_risk) + (0.1 * centrality)
        return np.clip(total_risk, 0, 1)

    def find_risk_clusters(self) -> List[List[str]]:
        """
        Identifies communities of entities that may share high risk.
        """
        if len(self.graph) < 2:
            return []
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(self.graph)
        return [list(c) for n, c in enumerate(communities)]

    def get_graph_context(self, entity_id: str) -> Dict[str, Any]:
        """
        Returns context about the entity's position in the graph for LLM reasoning.
        """
        if entity_id not in self.graph:
            return {}
            
        neighbors = list(self.graph.neighbors(entity_id))
        return {
            "degree": self.graph.degree(entity_id),
            "neighbors_count": len(neighbors),
            "connections": [
                {"entity": n, "type": self.graph[entity_id][n]['type']} 
                for n in neighbors[:5] # Limit context size
            ]
        }
