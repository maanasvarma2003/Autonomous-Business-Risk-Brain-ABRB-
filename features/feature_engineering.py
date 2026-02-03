import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List

class FeatureEngineer:
    """
    Standardizes feature engineering across different risk domains.
    """
    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def fit_transform_fraud(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Specific engineering for fraud detection.
        """
        # Data cleaning
        df = df.copy()
        
        # Identify numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in ['is_fraud', 'isFraud', 'TransactionID', 'TransactionDT']:
            if col in numeric_cols: numeric_cols.remove(col)
        
        # Handle missing values with better strategy
        df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()) if x.dtype != 'object' else x)
        
        # Add a few interaction features for better accuracy
        if 'TransactionAmt' in df.columns and 'dist1' in df.columns:
            df['amt_dist_ratio'] = df['TransactionAmt'] / (df['dist1'] + 1)
            numeric_cols.append('amt_dist_ratio')

        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.scalers['fraud'] = scaler
        return df, numeric_cols

    def fit_transform_churn(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Specific engineering for churn prediction.
        """
        df = df.copy()
        
        # Handle TotalCharges if present (often has string spaces)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            
        # Better Categorical encoding - using frequency encoding for high cardinality or simple Label for low
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        exclude = ['customerID', 'Churn', 'entity_id']
        cat_cols = [c for c in cat_cols if c not in exclude]
            
        for col in cat_cols:
            # For small cardinality, label encoding is fine
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[f'churn_{col}'] = le
            
        # Feature Scaling for Churn
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numeric_cols: numeric_cols.remove('Churn')
        
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.scalers['churn_scaler'] = scaler
            
        return df, numeric_cols

    def prepare_cyber_sequences(self, df: pd.DataFrame, window_size: int = 10) -> np.ndarray:
        """
        Prepares sequences for LSTM-based cyber anomaly detection.
        """
        # Simple sequence generation from numeric features
        numeric_data = df.select_dtypes(include=[np.number]).copy()
        
        # Clean infinities and NaNs which are common in network data
        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scaling is essential for LSTM
        scaler = StandardScaler()
        numeric_values = scaler.fit_transform(numeric_data.values)
        self.scalers['cyber'] = scaler
        
        sequences = []
        for i in range(len(numeric_values) - window_size):
            sequences.append(numeric_values[i:i+window_size])
        return np.array(sequences)
