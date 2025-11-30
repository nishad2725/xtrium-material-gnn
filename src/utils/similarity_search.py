"""Material similarity search utilities"""

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from typing import List, Dict, Optional


def find_similar_materials(material_idx: int,
                          top_k: int = 5,
                          embeddings: Optional[torch.Tensor] = None,
                          similarity_matrix: Optional[np.ndarray] = None,
                          df: Optional[pd.DataFrame] = None) -> List[Dict]:
    """
    Find top-k most similar materials to a given material.
    
    Args:
        material_idx: Index of the query material.
        top_k: Number of similar materials to return.
        embeddings: Learned material embeddings (PyTorch tensor).
        similarity_matrix: Pre-computed similarity matrix (numpy array).
        df: DataFrame with material information.
    
    Returns:
        List of dictionaries with material information and similarity scores.
    """
    if embeddings is not None:
        # Use learned embeddings
        material_embedding = embeddings[material_idx]
        similarities = F.cosine_similarity(
            material_embedding.unsqueeze(0),
            embeddings,
            dim=1
        ).numpy()
    elif similarity_matrix is not None:
        # Use original similarity matrix
        similarities = similarity_matrix[material_idx]
    else:
        raise ValueError("Either embeddings or similarity_matrix must be provided")
    
    # Get top-k (excluding self)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    results = []
    for idx in top_indices:
        result = {
            'material_idx': int(idx),
            'similarity': float(similarities[idx])
        }
        
        if df is not None:
            result['material_id'] = df.iloc[idx].get('Id', 'N/A')
            result['material_name'] = df.iloc[idx].get('Material_Name', 'N/A')
            result['category'] = df.iloc[idx].get('Category', 'N/A')
        
        results.append(result)
    
    return results

