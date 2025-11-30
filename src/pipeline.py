"""
Main pipeline script for Material GNN prototyping.

This script demonstrates how to use the modular code to:
1. Load and preprocess material data
2. Perform feature engineering
3. Build material similarity graph
4. Train GNN model
5. Perform similarity search
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import load_materials_data, get_column_types
from utils.feature_engineering import (
    create_property_completeness,
    encode_categorical_features,
    create_feature_matrix,
    normalize_features,
    create_node_features
)
from utils.graph_utils import build_material_graph, create_pyg_data, create_networkx_graph
from utils.similarity_search import find_similar_materials
from models.material_gnn import MaterialGNN
from training.trainer import MaterialGNNTrainer
import torch
import numpy as np


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("XTRIUM Material GNN Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/6] Loading materials data...")
    df = load_materials_data()
    print(f"Loaded {len(df)} materials with {len(df.columns)} features")
    
    # Step 2: Analyze column types
    print("\n[2/6] Analyzing data structure...")
    col_types = get_column_types(df)
    print(f"Numeric columns: {len(col_types['numeric'])}")
    print(f"Property value columns: {len(col_types['property_value'])}")
    print(f"Categorical columns: {len(col_types['categorical'])}")
    
    # Step 3: Feature engineering
    print("\n[3/6] Feature engineering...")
    df_fe = df.copy()
    
    # Property completeness
    property_completeness = create_property_completeness(df_fe, col_types['property_value'])
    df_fe['property_completeness'] = property_completeness
    print(f"Property completeness: mean={property_completeness.mean():.3f}")
    
    # Categorical encoding
    categorical_to_encode = ['Category', 'Database_Source', 'Manufacturer', 'Technical_Readiness_Level']
    categorical_to_encode = [c for c in categorical_to_encode if c in df_fe.columns]
    df_fe, label_encoders = encode_categorical_features(df_fe, categorical_to_encode)
    print(f"Encoded {len(categorical_to_encode)} categorical features")
    
    # Feature matrix
    feature_matrix, selected_properties = create_feature_matrix(
        df_fe, 
        col_types['property_value'],
        min_coverage=0.1,
        max_properties=20
    )
    print(f"Created feature matrix: {feature_matrix.shape}")
    print(f"Selected {len(selected_properties)} properties")
    
    # Normalize features
    feature_matrix_normalized, imputer, scaler = normalize_features(feature_matrix)
    print(f"Normalized features: mean={feature_matrix_normalized.mean():.4f}, std={feature_matrix_normalized.std():.4f}")
    
    # Node features
    node_features = create_node_features(
        feature_matrix_normalized,
        df_fe,
        categorical_to_encode,
        label_encoders,
        property_completeness
    )
    print(f"Node features shape: {node_features.shape}")
    
    # Step 4: Build graph
    print("\n[4/6] Building material similarity graph...")
    edges, edge_weights, similarity_matrix = build_material_graph(
        feature_matrix_normalized,
        similarity_threshold_percentile=80.0
    )
    print(f"Created graph with {len(edges)} edges")
    print(f"Average degree: {len(edges) * 2 / len(df_fe):.2f}")
    
    # Create PyTorch Geometric data
    data = create_pyg_data(node_features, edges, edge_weights)
    print(f"PyG Data: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_node_features} features")
    
    # Step 5: Initialize and train model
    print("\n[5/6] Initializing GNN model...")
    model = MaterialGNN(
        num_node_features=data.num_node_features,
        hidden_dim=64,
        num_layers=3,
        output_dim=32
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        print(f"Model output shape: {embeddings.shape}")
    
    # Optional: Train model (unsupervised for now)
    print("\nTraining model (unsupervised)...")
    trainer = MaterialGNNTrainer(model, learning_rate=0.01)
    history = trainer.train(
        train_data=data,
        num_epochs=50,
        verbose=True
    )
    
    # Step 6: Similarity search
    print("\n[6/6] Material similarity search...")
    final_embeddings = trainer.get_embeddings(data)
    final_embeddings_tensor = torch.tensor(final_embeddings, dtype=torch.float)
    
    test_material_idx = 0
    similar = find_similar_materials(
        test_material_idx,
        top_k=5,
        embeddings=final_embeddings_tensor,
        df=df_fe
    )
    
    print(f"\nQuery Material: {df_fe.iloc[test_material_idx]['Material_Name']}")
    print(f"\nTop 5 Similar Materials:")
    for i, result in enumerate(similar, 1):
        print(f"{i}. {result['material_name'][:60]}...")
        print(f"   Similarity: {result['similarity']:.4f}, Category: {result['category']}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return {
        'df': df_fe,
        'data': data,
        'model': model,
        'trainer': trainer,
        'embeddings': final_embeddings
    }


if __name__ == "__main__":
    results = main()

