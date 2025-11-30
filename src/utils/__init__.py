"""Utility modules for XTRIUM Material GNN project"""

from .data_utils import load_materials_data, find_repo_root
from .feature_engineering import (
    create_property_completeness,
    encode_categorical_features,
    create_feature_matrix,
    normalize_features
)
from .graph_utils import build_material_graph, create_pyg_data
from .similarity_search import find_similar_materials

__all__ = [
    'load_materials_data',
    'find_repo_root',
    'create_property_completeness',
    'encode_categorical_features',
    'create_feature_matrix',
    'normalize_features',
    'build_material_graph',
    'create_pyg_data',
    'find_similar_materials',
]

