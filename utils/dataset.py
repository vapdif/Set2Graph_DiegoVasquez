import os
import uproot
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

# Definimos las características de los nodos y jets
node_features_list = ['trk_d0', 'trk_z0', 'trk_phi', 'trk_ctgtheta', 'trk_pt', 'trk_charge']
jet_features_list = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_M']

class JetGraphDataset(Dataset):
    def __init__(self, which_set, data_dir="data/", debug_load=True):
        """
        Inicialización del dataset
        :param which_set: 'train', 'validation' o 'test'
        :param data_dir: Ruta base de los datos
        :param debug_load: Si True, cargará un subconjunto pequeño para depuración
        """
        assert which_set in ['train', 'validation', 'test']
        fname = {'train': 'training_data.root', 'validation': 'valid_data.root', 'test': 'test_data.root'}
        self.filename = os.path.join(data_dir, which_set, fname[which_set])

        # Cargar datos del archivo ROOT
        with uproot.open(self.filename) as f:
            tree = f['tree']  # Cambiar según la clave válida
            self.n_jets = int(tree.num_entries)  # Número total de jets
            self.jet_arrays = tree.arrays(jet_features_list + node_features_list + ['trk_vtx_index'])

        if debug_load:
            self.n_jets = 1000 # Cargar un subconjunto pequeño para depuración

    def __len__(self):
        """
        Devuelve el número total de jets en el dataset.
        """
        return self.n_jets

    def __getitem__(self, idx):
        # Extraer características del jet y nodos
        jet_features = [self.jet_arrays[feature][idx] for feature in jet_features_list]
        node_features = [np.array(self.jet_arrays[feature][idx]) for feature in node_features_list]
        node_labels = np.array(self.jet_arrays['trk_vtx_index'][idx])

        n_nodes = len(node_labels)
        if not all(len(feature) == n_nodes for feature in node_features):
            raise ValueError(f"Inconsistent node features for jet {idx}. Expected {n_nodes}, but got {[len(f) for f in node_features]}.")

        # Convertir características de los nodos a un arreglo de NumPy
        node_feats = np.vstack(node_features).T  # (n_nodes, n_features)

        # Crear características de jet y replicarlas para todos los nodos
        jet_feats = np.array(jet_features).reshape(1, -1)  # (1, jet_features_dim)
        jet_feats = np.repeat(jet_feats, n_nodes, axis=0)  # Repetir para todos los nodos

        # Concatenar características de nodos y jets
        x = torch.tensor(np.hstack([node_feats, jet_feats]), dtype=torch.float)

        # Construir el grafo inicial conectando nodos del mismo vértice
        edge_index = []
        edge_labels = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edge_index.append((i, j))
                edge_index.append((j, i))
                label = 1 if node_labels[i] == node_labels[j] else 0
                edge_labels.append(label)
                edge_labels.append(label)

        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # (2, num_edges)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float)  # Etiquetas de los bordes

        return Data(x=x, edge_index=edge_index, y=edge_labels)




"""def collate_fn(data_list):
    max_nodes = max([data.x.shape[0] for data in data_list])
    print(f"Max nodes in batch: {max_nodes}")
    
    for data in data_list:
        n_nodes = data.x.shape[0]
        print(f"Data nodes: {n_nodes}")
        if n_nodes < max_nodes:
            padding = torch.zeros((max_nodes - n_nodes, data.x.shape[1]), dtype=data.x.dtype)
            data.x = torch.cat([data.x, padding], dim=0)

    return Batch.from_data_list(data_list)"""


