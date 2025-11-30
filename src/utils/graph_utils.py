"""Graph construction utilities for material similarity graphs"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional


def build_material_graph(feature_matrix: np.ndarray,
                        similarity_threshold_percentile: float = 80.0) -> Tuple[List[Tuple[int, int]], List[float], np.ndarray]:
    """
    Build material similarity graph based on feature similarity.
    
    Args:
        feature_matrix: Normalized feature matrix.
        similarity_threshold_percentile: Percentile threshold for edge creation.
    
    Returns:
        Tuple of (edge list, edge weights, similarity matrix).
    """
    # Calculate material similarity using cosine similarity
    similarity_matrix = cosine_similarity(feature_matrix)
    
    # Set similarity threshold for edges
    similarity_threshold = np.percentile(similarity_matrix[similarity_matrix < 1.0], 
                                        similarity_threshold_percentile)
    
    # Create edge list
    edges = []
    edge_weights = []
    for i in range(len(feature_matrix)):
        for j in range(i + 1, len(feature_matrix)):
            if similarity_matrix[i, j] > similarity_threshold:
                edges.append((i, j))
                edge_weights.append(similarity_matrix[i, j])
    
    return edges, edge_weights, similarity_matrix


def create_networkx_graph(num_nodes: int, edges: List[Tuple[int, int]]) -> nx.Graph:
    """
    Create NetworkX graph from edge list.
    
    Args:
        num_nodes: Number of nodes in the graph.
        edges: List of edge tuples.
    
    Returns:
        NetworkX Graph object.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    return G


def create_pyg_data(node_features: np.ndarray,
                   edges: List[Tuple[int, int]],
                   edge_weights: List[float]) -> Data:
    """
    Create PyTorch Geometric Data object from node features and edges.
    
    Args:
        node_features: Node feature matrix.
        edges: List of edge tuples.
        edge_weights: List of edge weights.
    
    Returns:
        PyTorch Geometric Data object.
    """
    # Convert edge list to PyTorch format (bidirectional for undirected graph)
    edge_list_forward = [[e[0] for e in edges], [e[1] for e in edges]]
    edge_list_reverse = [[e[1] for e in edges], [e[0] for e in edges]]
    
    # Concatenate properly: [[src1, src2, ...], [dst1, dst2, ...]]
    edge_index = torch.tensor([
        edge_list_forward[0] + edge_list_reverse[0],
        edge_list_forward[1] + edge_list_reverse[1]
    ], dtype=torch.long)
    
    # Convert node features to tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create edge attributes (weights) - duplicate for bidirectional
    edge_attr = torch.tensor(edge_weights + edge_weights, dtype=torch.float).unsqueeze(1)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

