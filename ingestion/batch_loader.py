import kagglehub
import pandas as pd
import os
import sys
import shutil
from typing import Dict

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BatchLoader:
    """
    Handles downloading and preparing datasets for ABRB.
    """
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_all(self) -> Dict[str, str]:
        """
        Downloads all mandatory datasets using provided kagglehub code.
        """
        datasets = {
            "fraud": "antonsruberts/fraud-detection-preprocesed",
            "cyber": "chethuhn/network-intrusion-dataset",
            "compliance_gdpr": "jessemostipak/gdpr-violations",
            "compliance_ledger": "chrissyserb/sovereign-ledger-protocol",
            "churn": "yeanzc/telco-customer-churn-ibm-dataset"
        }
        
        paths = {}
        for key, repo in datasets.items():
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Executing (Attempt {attempt+1}): path = kagglehub.dataset_download('{repo}')")
                    path = kagglehub.dataset_download(repo)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Final attempt failed for {key}: {e}")
                        raise e
                    print(f"Attempt {attempt+1} failed for {key}, retrying...")
            
            dest = os.path.join(self.data_dir, key)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            
            # If path is a file, create directory and move file
            if os.path.isfile(path):
                os.makedirs(dest, exist_ok=True)
                shutil.copy(path, os.path.join(dest, os.path.basename(path)))
            else:
                shutil.copytree(path, dest)
            
            paths[key] = dest
            print(f"Path to {key} dataset files: {dest}")
            
        return paths

    def load_dataset(self, name: str) -> pd.DataFrame:
        """
        Loads a specific dataset into a pandas DataFrame.
        """
        path = os.path.join(self.data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset directory not found: {path}")

        # Find all data files in the directory and subdirectories
        data_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(('.csv', '.xlsx', '.parquet')):
                    data_files.append(os.path.join(root, file))

        if not data_files:
            raise FileNotFoundError(f"No data files found in {path}")
        
        # Load the largest file (usually the main data)
        data_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
        target_file = data_files[0]
        
        if target_file.endswith('.csv'):
            return pd.read_csv(target_file, low_memory=False)
        elif target_file.endswith('.xlsx'):
            return pd.read_excel(target_file)
        elif target_file.endswith('.parquet'):
            return pd.read_parquet(target_file)
        
        return pd.read_csv(target_file, low_memory=False)

if __name__ == "__main__":
    loader = BatchLoader()
    try:
        paths = loader.download_all()
        print("All datasets downloaded successfully.")
    except Exception as e:
        print(f"Error downloading datasets: {e}")
        # In a real production scenario, we might fallback to local mock data
