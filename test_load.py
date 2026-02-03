import pandas as pd
import os

path = 'datasets/fraud/fraud_train_preprocessed.csv'
if os.path.exists(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path, nrows=5)
    print("Columns:", df.columns.tolist())
    print("Success")
else:
    print(f"File {path} not found")
