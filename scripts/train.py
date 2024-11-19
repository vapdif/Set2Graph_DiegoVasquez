from models.set2graph_model import Set2GraphModel
from utils.dataset import JetsDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(data_path, model_path, batch_size=64, input_dim=10, hidden_dim=128, output_dim=1, num_epochs=20, lr=1e-3):
    train_dataset = JetsDataset(f"{data_path}/train/training_data.root")
    val_dataset = JetsDataset(f"{data_path}/validation/valid_data.root")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Set2GraphModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for tracks, labels in train_loader:
            tracks, labels = tracks.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tracks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
