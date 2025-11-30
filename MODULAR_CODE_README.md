# Modular Code Structure

This document describes the modular code structure created from the notebook analysis.

## Directory Structure

```
src/
├── models/
│   ├── __init__.py
│   ├── material_gnn.py          # GNN model architectures
│   └── hetero_gnn.py           # (Placeholder for future heterogeneous GNN)
│
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training utilities and trainer class
│   └── train_link_pred.py       # (Placeholder for link prediction training)
│
├── utils/
│   ├── __init__.py
│   ├── data_utils.py            # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature engineering functions
│   ├── graph_utils.py           # Graph construction utilities
│   └── similarity_search.py     # Material similarity search
│
└── pipeline.py                  # Main pipeline script

```

## Module Descriptions

### `src/utils/data_utils.py`
- `load_materials_data()`: Load materials CSV file
- `find_repo_root()`: Find repository root directory
- `get_column_types()`: Categorize columns (numeric, categorical, text, property_value)

### `src/utils/feature_engineering.py`
- `create_property_completeness()`: Calculate property completeness scores
- `encode_categorical_features()`: Encode categorical features using LabelEncoder
- `create_feature_matrix()`: Create feature matrix from material properties
- `normalize_features()`: Normalize features with imputation and scaling
- `create_node_features()`: Combine all features into node feature vectors

### `src/utils/graph_utils.py`
- `build_material_graph()`: Build similarity graph from feature matrix
- `create_networkx_graph()`: Create NetworkX graph for visualization
- `create_pyg_data()`: Convert to PyTorch Geometric Data format

### `src/utils/similarity_search.py`
- `find_similar_materials()`: Find top-k similar materials using embeddings

### `src/models/material_gnn.py`
- `MaterialGNN`: Base GNN model for material embeddings
- `MaterialGNNPredictor`: GNN with property prediction head

### `src/training/trainer.py`
- `MaterialGNNTrainer`: Training class with train/validate loops, embedding extraction, model save/load

### `src/pipeline.py`
Complete end-to-end pipeline demonstrating all modules working together.

## Usage Example

```python
from src.utils import load_materials_data, create_feature_matrix, normalize_features
from src.models import MaterialGNN
from src.training import MaterialGNNTrainer

# Load data
df = load_materials_data()

# Feature engineering
feature_matrix, selected_props = create_feature_matrix(df, property_cols)
feature_matrix_norm, _, _ = normalize_features(feature_matrix)

# Build graph and create PyG data
edges, weights, _ = build_material_graph(feature_matrix_norm)
data = create_pyg_data(node_features, edges, weights)

# Initialize model
model = MaterialGNN(num_node_features=data.num_node_features)

# Train
trainer = MaterialGNNTrainer(model)
trainer.train(train_data=data, num_epochs=100)

# Get embeddings
embeddings = trainer.get_embeddings(data)
```

## Virtual Environment

A virtual environment has been created at `venv/` with all required packages installed.

To activate:
```bash
source venv/bin/activate
```

Jupyter kernel has been registered as `xtrium-gnn`. Select this kernel when running the notebook.

## Running the Pipeline

```bash
# Activate virtual environment
source venv/bin/activate

# Run the pipeline
python src/pipeline.py
```

## Running the Notebook

1. Activate virtual environment: `source venv/bin/activate`
2. Start Jupyter: `jupyter lab`
3. Open `notebooks/02_GNN_Prototype_EDA.ipynb`
4. Select kernel: `Python (xtrium-gnn)`
5. Run all cells

## Next Steps

1. **Model Training**: Implement supervised training for property prediction
2. **Evaluation Metrics**: Add comprehensive evaluation metrics
3. **Hyperparameter Tuning**: Implement hyperparameter optimization
4. **Production Pipeline**: Integrate with XTRIUM platform
5. **Link Prediction**: Implement link prediction training (see `train_link_pred.py`)

