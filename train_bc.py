import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 6)
        )
    def forward(self, x):
        return self.net(x)

class OvercookedDataset(Dataset):
    def __init__(self, data_dir):
        self.obs = []
        self.actions = []
        
        npz_files = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
        print(f"Found {len(npz_files)} demonstration files.")
        
        for file_path in npz_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                self.obs.append(data["obs"])
                self.actions.append(data["actions"])
            except Exception as e:
                print(f"Skipping corrupt file {file_path}: {e}")
                
        if self.obs:
            self.obs = np.concatenate(self.obs, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            print(f"Loaded total {len(self.obs)} state-action pairs.")
        else:
            print("No observations loaded.")
            self.obs = np.zeros((0, 96), dtype=np.float32)
            self.actions = np.zeros((0,), dtype=np.int64)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.obs[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.long)
        )

def main():
    data_dir = "./data"
    os.makedirs("models", exist_ok=True)
    
    print("Loading datasets...")
    dataset = OvercookedDataset(data_dir)
    if len(dataset) == 0:
        print("Error: No data available for Behavior Cloning.")
        return
        
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = PolicyNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 40
    print(f"Starting Behavior Cloning training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for obs_batch, act_batch in loader:
            optimizer.zero_grad()
            outputs = model(obs_batch)
            loss = criterion(outputs, act_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * obs_batch.size(0)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == act_batch).sum().item()
            total += obs_batch.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
        
    save_path = "models/ppo_general_agent.zip"
    torch.save(model.state_dict(), save_path)
    print(f"Successfully trained general BC agent! Saved model weights to {save_path}")

if __name__ == "__main__":
    main()
