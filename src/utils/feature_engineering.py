"""Feature engineering utilities for material data"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import List, Dict, Optional, Tuple


def create_property_completeness(df: pd.DataFrame, property_value_cols: List[str]) -> pd.Series:
    """
    Create property completeness score for each material.
    
    Args:
        df: Input DataFrame.
        property_value_cols: List of property value column names.
    
    Returns:
        Series with completeness scores (0-1).
    """
    property_value_cols_available = [col for col in property_value_cols if col in df.columns]
    completeness = df[property_value_cols_available].notna().sum(axis=1) / len(property_value_cols_available)
    return completeness


def encode_categorical_features(df: pd.DataFrame, 
                               categorical_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df: Input DataFrame.
        categorical_cols: List of categorical column names to encode.
    
    Returns:
        Tuple of (DataFrame with encoded columns, dictionary of label encoders).
    """
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            # Fill NaN with 'Unknown' before encoding
            df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].fillna('Unknown'))
            label_encoders[col] = le
    
    return df_encoded, label_encoders


def create_feature_matrix(df: pd.DataFrame, 
                         property_value_cols: List[str],
                         min_coverage: float = 0.1,
                         max_properties: int = 20) -> Tuple[np.ndarray, List[str]]:
    """
    Create feature matrix from material properties.
    
    Args:
        df: Input DataFrame.
        property_value_cols: List of property value column names.
        min_coverage: Minimum coverage threshold (0-1) for property selection.
        max_properties: Maximum number of properties to select.
    
    Returns:
        Tuple of (feature matrix, list of selected property names).
    """
    # Calculate property coverage
    property_coverage = {}
    for col in property_value_cols:
        if col in df.columns:
            coverage = df[col].notna().sum() / len(df)
            property_coverage[col] = coverage
    
    # Select properties with at least min_coverage coverage
    selected_properties = [prop for prop, cov in property_coverage.items() if cov >= min_coverage]
    selected_properties = sorted(selected_properties, 
                                key=lambda x: property_coverage[x], 
                                reverse=True)[:max_properties]
    
    # Create feature matrix (fill missing with 0 for now)
    feature_matrix = df[selected_properties].fillna(0).values
    
    return feature_matrix, selected_properties


def normalize_features(feature_matrix: np.ndarray, 
                       imputation_strategy: str = 'median') -> Tuple[np.ndarray, SimpleImputer, StandardScaler]:
    """
    Normalize features using imputation and standardization.
    
    Args:
        feature_matrix: Input feature matrix.
        imputation_strategy: Strategy for imputation ('median', 'mean', 'most_frequent').
    
    Returns:
        Tuple of (normalized feature matrix, fitted imputer, fitted scaler).
    """
    # Impute missing values
    imputer = SimpleImputer(strategy=imputation_strategy)
    feature_matrix_imputed = imputer.fit_transform(feature_matrix)
    
    # Normalize features
    scaler = StandardScaler()
    feature_matrix_normalized = scaler.fit_transform(feature_matrix_imputed)
    
    return feature_matrix_normalized, imputer, scaler


def create_node_features(feature_matrix_normalized: np.ndarray,
                        df: pd.DataFrame,
                        categorical_cols: List[str],
                        label_encoders: Dict[str, LabelEncoder],
                        property_completeness: pd.Series) -> np.ndarray:
    """
    Create node features by combining property features, categorical encodings, and completeness.
    
    Args:
        feature_matrix_normalized: Normalized property feature matrix.
        df: DataFrame with encoded categorical features.
        categorical_cols: List of categorical column names.
        label_encoders: Dictionary of label encoders.
        property_completeness: Property completeness scores.
    
    Returns:
        Node feature matrix.
    """
    node_features_list = []
    
    for idx in range(len(df)):
        # Start with normalized property features
        features = list(feature_matrix_normalized[idx])
        
        # Add categorical encodings (normalized)
        for col in categorical_cols:
            if col + '_encoded' in df.columns:
                encoded_val = df[col + '_encoded'].iloc[idx]
                # Normalize encoded value
                features.append(encoded_val / len(label_encoders[col].classes_))
        
        # Add property completeness
        features.append(property_completeness.iloc[idx])
        
        node_features_list.append(features)
    
    return np.array(node_features_list)

