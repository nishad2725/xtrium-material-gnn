"""Graph Neural Network model for material property prediction and similarity"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import Optional


class MaterialGNN(torch.nn.Module):
    """
    Graph Neural Network for Material Property Prediction and Embedding.
    
    This model uses Graph Convolutional Networks (GCN) to learn material
    representations based on material properties and similarity relationships.
    """
    
    def __init__(self, 
                 num_node_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 output_dim: int = 32,
                 dropout: float = 0.5,
                 use_gat: bool = False):
        """
        Initialize MaterialGNN model.
        
        Args:
            num_node_features: Number of input node features.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of GNN layers.
            output_dim: Dimension of output embeddings.
            dropout: Dropout probability.
            use_gat: If True, use GAT instead of GCN.
        """
        super(MaterialGNN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.use_gat = use_gat
        
        ConvLayer = GATConv if use_gat else GCNConv
        
        # First layer
        if use_gat:
            self.convs.append(ConvLayer(num_node_features, hidden_dim, heads=1))
        else:
            self.convs.append(ConvLayer(num_node_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if use_gat:
                self.convs.append(ConvLayer(hidden_dim, hidden_dim, heads=1))
            else:
                self.convs.append(ConvLayer(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            if use_gat:
                self.convs.append(ConvLayer(hidden_dim, output_dim, heads=1))
            else:
                self.convs.append(ConvLayer(hidden_dim, output_dim))
        else:
            if use_gat:
                self.convs.append(ConvLayer(num_node_features, output_dim, heads=1))
            else:
                self.convs.append(ConvLayer(num_node_features, output_dim))
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node feature matrix [num_nodes, num_node_features].
            edge_index: Graph connectivity [2, num_edges].
        
        Returns:
            Node embeddings [num_nodes, output_dim].
        """
        # Graph convolution layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer (no activation for embedding)
        x = self.convs[-1](x, edge_index)
        
        return x


class MaterialGNNPredictor(torch.nn.Module):
    """
    Material property predictor that extends MaterialGNN with prediction head.
    """
    
    def __init__(self,
                 num_node_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 embedding_dim: int = 32,
                 num_properties: int = 1,
                 dropout: float = 0.5):
        """
        Initialize MaterialGNNPredictor.
        
        Args:
            num_node_features: Number of input node features.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of GNN layers.
            embedding_dim: Dimension of material embeddings.
            num_properties: Number of properties to predict.
            dropout: Dropout probability.
        """
        super(MaterialGNNPredictor, self).__init__()
        
        self.gnn = MaterialGNN(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        # Prediction head
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_properties)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: GNN embedding + property prediction.
        
        Args:
            x: Node feature matrix.
            edge_index: Graph connectivity.
        
        Returns:
            Property predictions [num_nodes, num_properties].
        """
        embeddings = self.gnn(x, edge_index)
        predictions = self.predictor(embeddings)
        return predictions
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get material embeddings without prediction.
        
        Args:
            x: Node feature matrix.
            edge_index: Graph connectivity.
        
        Returns:
            Material embeddings.
        """
        return self.gnn(x, edge_index)

