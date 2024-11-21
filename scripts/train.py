from models.set2graph_model import Set2GraphModel  # Importar el modelo
from utils.dataset import JetGraphDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score


import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(data_path, model_path, batch_size=64, input_dim=10, hidden_dim=128, output_dim=1, num_epochs=100, lr=1e-3):
    total_positive = 0
    total_negative = 0

    # Cargar datasets
    train_dataset = JetGraphDataset('train', data_dir=data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Inicializar modelo, pérdida y optimizador
    model = Set2GraphModel(input_dim, hidden_dim, output_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()          # funcion de perdida     cambiada - nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    
    # Entrenamiento
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            total_positive += data.y.sum().item()
            total_negative += (data.y.size(0) - data.y.sum().item())

            #print(f"Total bordes positivos: {total_positive}")
            #print(f"Total bordes negativos: {total_negative}")

            data = data.to(DEVICE)
            output = model(data.x, data.edge_index)
            labels = data.y.to(DEVICE)

            # Asegurar que las dimensiones coincidan
            output = output.view(-1)
            labels = labels.view(-1)

            # Calcular la pérdida
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss / len(train_loader)}")

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")
