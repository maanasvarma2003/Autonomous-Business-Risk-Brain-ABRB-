import pandas as pd
import numpy as np
import os

def generate_mock_fraud_data(n_samples: int = 1000):
    df = pd.DataFrame({
        'amount': np.random.uniform(1, 10000, n_samples),
        'v1': np.random.randn(n_samples),
        'v2': np.random.randn(n_samples),
        'v3': np.random.randn(n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })
    return df

def generate_mock_churn_data(n_samples: int = 1000):
    df = pd.DataFrame({
        'tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(20, 8000, n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    return df

def generate_mock_cyber_data(n_samples: int = 1000):
    # Network logs: duration, protocol_type, service, src_bytes, dst_bytes
    df = pd.DataFrame({
        'duration': np.random.exponential(1, n_samples),
        'src_bytes': np.random.randint(0, 100000, n_samples),
        'dst_bytes': np.random.randint(0, 100000, n_samples),
        'count': np.random.randint(0, 500, n_samples)
    })
    return df

def generate_mock_compliance_data(n_samples: int = 1000):
    policy_texts = [
        "User accessed confidential database without MFA",
        "Transaction exceeds threshold and lacks approval",
        "Normal login from recognized device",
        "Updated password according to policy",
        "Exported user data to unauthorized external IP",
        "Audit log cleaned up by administrator"
    ]
    labels = [1, 1, 0, 0, 1, 1]
    
    texts = [np.random.choice(policy_texts) for _ in range(n_samples)]
    # Match labels to the chosen text (simplified)
    label_map = {text: label for text, label in zip(policy_texts, labels)}
    sample_labels = [label_map[t] for t in texts]
    
    return pd.DataFrame({'text': texts, 'violation': sample_labels})

def ensure_datasets_exist():
    """
    Creates mock CSVs if kaggle downloads are missing.
    """
    base = "datasets"
    if not os.path.exists(base): os.makedirs(base)
    
    if not os.path.exists(f"{base}/fraud"):
        os.makedirs(f"{base}/fraud")
        generate_mock_fraud_data().to_csv(f"{base}/fraud/mock_fraud.csv", index=False)
        
    if not os.path.exists(f"{base}/churn"):
        os.makedirs(f"{base}/churn")
        generate_mock_churn_data().to_csv(f"{base}/churn/mock_churn.csv", index=False)
        
    if not os.path.exists(f"{base}/cyber"):
        os.makedirs(f"{base}/cyber")
        generate_mock_cyber_data().to_csv(f"{base}/cyber/mock_cyber.csv", index=False)
        
    if not os.path.exists(f"{base}/compliance"):
        os.makedirs(f"{base}/compliance")
        generate_mock_compliance_data().to_csv(f"{base}/compliance/mock_compliance.csv", index=False)
