import torch
import torch.nn as nn
import numpy as np
import os

class CyberLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(CyberLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

class CyberModel:
    """
    Cyber Risk Model: LSTM-based autoencoder for sequence anomaly detection.
    Detects deviations from normal login/network patterns.
    """
    def __init__(self, input_dim: int, model_dir: str = "models/cyber"):
        self.input_dim = input_dim
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CyberLSTM(input_dim).to(self.device)
        self.threshold = 0.01 # Reconstruct error threshold

    def train(self, sequences: np.ndarray, epochs: int = 30, batch_size: int = 128):
        print(f"Training Cyber LSTM Model on {len(sequences)} sequences...")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Mini-batch training for better stability/accuracy
        num_batches = len(sequences) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            # Shuffle indices
            indices = np.arange(len(sequences))
            np.random.shuffle(indices)
            
            for i in range(0, len(sequences), batch_size):
                batch_idx = indices[i:i+batch_size]
                if len(batch_idx) < batch_size // 2: continue
                
                batch_data = torch.FloatTensor(sequences[batch_idx]).to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_data)
                loss = criterion(output, batch_data[:, -1, :])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 5 == 0:
                avg_loss = epoch_loss / (num_batches + 1)
                print(f"Epoch {epoch}, Avg Loss: {avg_loss:.8f}")
        
        # Calculate adaptive threshold with full data (eval mode)
        self.model.eval()
        with torch.no_grad():
            all_data = torch.FloatTensor(sequences).to(self.device)
            preds = self.model(all_data)
            errors = torch.mean((preds - all_data[:, -1, :])**2, dim=1).cpu().numpy()
            # 99.5th percentile for better precision in anomaly detection
            self.threshold = np.percentile(errors, 99.5)
            print(f"Cyber threshold set at: {self.threshold:.8f}")
            
        self.save_model()
        print("Cyber Model training complete.")

    def predict_risk(self, sequences: np.ndarray) -> np.ndarray:
        self.model.eval()
        data = torch.FloatTensor(sequences).to(self.device)
        with torch.no_grad():
            preds = self.model(data)
            errors = torch.mean((preds - data[:, -1, :])**2, dim=1).cpu().numpy()
            
        # Normalize error to [0,1] risk score
        risk_scores = np.clip(errors / (self.threshold * 2), 0, 1)
        return risk_scores

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "cyber_lstm.pth"))
        np.save(os.path.join(self.model_dir, "threshold.npy"), self.threshold)

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, "cyber_lstm.pth")))
        self.threshold = np.load(os.path.join(self.model_dir, "threshold.npy"))
