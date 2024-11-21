from utils.dataset import JetGraphDataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np

from models.set2graph_model import Set2GraphModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix, balanced_accuracy_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusión', cmap=plt.cm.Blues):
    """
    Esta función imprime y grafica la matriz de confusión.
    Normalización se puede aplicar configurando `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusión Normalizada")
    else:
        print('Matriz de Confusión, sin normalizar')

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_prob, title='Curva ROC'):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def evaluate_model(data_path, model_path, batch_size=64, input_dim=10, hidden_dim=128, output_dim=1):
    # Cargar dataset de prueba
    test_dataset = JetGraphDataset('test', data_dir=data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Cargar modelo
    model = Set2GraphModel(input_dim, hidden_dim, output_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluar
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            output = model(data.x, data.edge_index)
            probabilities = torch.sigmoid(output)
            predictions = (probabilities > 0.5).long()
            y_true.append(data.y.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())
            y_prob.append(probabilities.cpu().numpy())

    # Concatenar todas las predicciones y etiquetas
    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)
    y_prob = np.hstack(y_prob)

    # Calcular métricas
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float('nan')  # Manejar casos donde no se puede calcular
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Imprimir métricas
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Definir nombres de clases
    classes = ['No Same Vertex', 'Same Vertex']  # Ajusta según tus etiquetas

    # Graficar Matriz de Confusión
    plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix')

    # Graficar Curva ROC
    if not np.isnan(roc_auc):
        plot_roc_curve(y_true, y_prob, title='ROC Curve')






