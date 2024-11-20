from utils.dataset import JetGraphDataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np

from models.set2graph_model import Set2GraphModel
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(data_path, model_path, batch_size=64, input_dim=10, hidden_dim=128, output_dim=1):
    # Cargar dataset de prueba
    test_dataset = JetGraphDataset('test', data_dir=data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Cargar modelo
    model = Set2GraphModel(input_dim, hidden_dim, output_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluar
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            output = model(data.x, data.edge_index)
            probabilities = torch.sigmoid(output)
            predictions = (probabilities > 0.5).long()
            #predictions = (output > 0.5).long()
            y_true.append(data.y.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())

    # Calcular métricas
    f1 = f1_score(np.hstack(y_true), np.hstack(y_pred), average='weighted')
    print(f"F1 Score: {f1}")

