# XTRIUM Material GNN

Graph Neural Network project for material discovery and property prediction using AI/ML solutions.

## About

This project implements a Graph Neural Network (GNN) prototyping model for material discovery at XTRIUM, an AI-driven platform that uses ML/AI solutions for material innovation. The platform aims to:

- **Match Materials to Applications**: Identify novel and non-obvious applications for existing materials
- **Identify Substitutes**: Find cost-effective, sustainable, and geographically diverse alternative materials
- **Optimize Supply Chains**: Provide supply chain visibility and connect buyers and sellers
- **Drive Sustainability**: Track circularity, CO₂ footprint, and certifications

## Project Structure

```
xtrium-material-gnn/
├── data/
│   ├── raw/
│   │   └── materials.csv          # Material properties dataset
│   └── processed/                  # Processed data (generated)
├── notebooks/
│   ├── 01_EDA_materials.ipynb     # Initial exploratory data analysis
│   └── 02_GNN_Prototype_EDA.ipynb # Comprehensive EDA and GNN prototyping
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── material_gnn.py        # GNN model architectures
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py             # Training utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_utils.py          # Data loading utilities
│   │   ├── feature_engineering.py # Feature engineering functions
│   │   ├── graph_utils.py         # Graph construction utilities
│   │   └── similarity_search.py   # Material similarity search
│   └── pipeline.py                # End-to-end pipeline script
├── configs/
│   └── config.yaml                # Configuration file
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment (optional)
└── README.md                      # This file
```

## Features

- **Comprehensive EDA**: Exploratory data analysis with missing value handling, feature engineering, and visualizations
- **Graph Construction**: Build material similarity graphs based on property values
- **GNN Models**: Graph Convolutional Network (GCN) models for material embedding and property prediction
- **Material Similarity Search**: Find similar materials using learned embeddings
- **Modular Codebase**: Clean, modular code structure for easy extension and maintenance

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/nishad2725/xtrium-material-gnn.git
cd xtrium-material-gnn
```

2. **Create virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Register Jupyter kernel** (optional, for notebook execution):
```bash
pip install ipykernel
python -m ipykernel install --user --name=xtrium-gnn --display-name "Python (xtrium-gnn)"
```

## Usage

### Running the Pipeline

Execute the complete pipeline:
```bash
python src/pipeline.py
```

### Running the Notebook

1. Start Jupyter Lab:
```bash
jupyter lab
```

2. Open `notebooks/02_GNN_Prototype_EDA.ipynb`
3. Select kernel: `Python (xtrium-gnn)`
4. Run all cells

### Using the Modular Code

```python
from src.utils import load_materials_data, create_feature_matrix, normalize_features
from src.models import MaterialGNN
from src.training import MaterialGNNTrainer
from src.utils.graph_utils import build_material_graph, create_pyg_data

# Load data
df = load_materials_data()

# Feature engineering
feature_matrix, selected_props = create_feature_matrix(df, property_cols)
feature_matrix_norm, _, _ = normalize_features(feature_matrix)

# Build graph
edges, weights, similarity_matrix = build_material_graph(feature_matrix_norm)
data = create_pyg_data(node_features, edges, weights)

# Initialize and train model
model = MaterialGNN(num_node_features=data.num_node_features)
trainer = MaterialGNNTrainer(model)
trainer.train(train_data=data, num_epochs=100)

# Get embeddings
embeddings = trainer.get_embeddings(data)
```

## Model Architecture

- **Input**: Node features combining normalized property values + categorical encodings
- **Architecture**: Multi-layer Graph Convolutional Network (GCN)
- **Output**: 32-dimensional material embeddings
- **Graph**: Undirected graph with similarity-based edges

## Key Modules

### `src/utils/data_utils.py`
- Data loading and preprocessing
- Column type analysis

### `src/utils/feature_engineering.py`
- Property completeness scoring
- Categorical encoding
- Feature matrix creation
- Feature normalization

### `src/utils/graph_utils.py`
- Material similarity calculation
- Graph construction
- PyTorch Geometric data conversion

### `src/models/material_gnn.py`
- `MaterialGNN`: Base GNN for embeddings
- `MaterialGNNPredictor`: GNN with property prediction head

### `src/training/trainer.py`
- Training loops
- Model evaluation
- Embedding extraction
- Model save/load

## Data

The dataset (`data/raw/materials.csv`) contains material properties including:
- Physical properties (density, thermal conductivity, specific heat)
- Mechanical properties (tensile strength, modulus, yield strength)
- Thermal properties (coefficient of thermal expansion, temperature limits)
- Material metadata (category, manufacturer, database source)

## Development

### Code Structure

The codebase follows a modular structure:
- **Utils**: Reusable utility functions
- **Models**: GNN model definitions
- **Training**: Training and evaluation code
- **Pipeline**: End-to-end execution scripts

### Adding New Features

1. Add utility functions to `src/utils/`
2. Add model architectures to `src/models/`
3. Add training scripts to `src/training/`
4. Update `src/pipeline.py` to integrate new features

## Requirements

See `requirements.txt` for full list. Key dependencies:
- torch>=2.1.0
- torch-geometric
- pandas
- scikit-learn
- networkx
- matplotlib
- seaborn
- jupyterlab

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Contact

For questions or issues, please open an issue on GitHub.

## References

- XTRIUM: AI Materials Platform for material innovation
- PyTorch Geometric: Graph Neural Network library
- NASA Langley Research Center Database: Material properties data source
