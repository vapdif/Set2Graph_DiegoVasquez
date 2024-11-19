from models.set2graph_model import Set2GraphModel
from utils.dataset import JetsDataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, adjusted_rand_score

def evaluate_model(data_path, model_path, batch_size=64, input_dim=10, hidden_dim=128, output_dim=1):
    test_dataset = JetsDataset(f"{data_path}/test/test_data.root")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Set2GraphModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for tracks, labels in test_loader:
            tracks = tracks.to(device)
            preds = (model(tracks) > 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    ari = adjusted_rand_score(all_labels.flatten(), all_preds.flatten())
    print(f"F1 Score: {f1}, ARI: {ari}")
