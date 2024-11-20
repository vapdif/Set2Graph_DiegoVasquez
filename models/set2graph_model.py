import torch
import torch.nn as nn

class Set2GraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Set2GraphModel, self).__init__()
        self.set_to_set = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim),
            #nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        # Obtener embeddings de los nodos
        track_embeddings = self.set_to_set(x)  # (num_nodes_total, hidden_dim)

        # Obtener embeddings de los nodos para los bordes
        sender_embeddings = track_embeddings[edge_index[0]]  # (num_edges_total, hidden_dim)
        receiver_embeddings = track_embeddings[edge_index[1]]  # (num_edges_total, hidden_dim)

        # Concatenar embeddings de los nodos para formar las características de los bordes
        edge_features = torch.cat([sender_embeddings, receiver_embeddings], dim=-1)  # (num_edges_total, 2 * hidden_dim)

        # Clasificar los bordes
        edge_scores = self.edge_classifier(edge_features).squeeze(-1)  # (num_edges_total)

        return edge_scores

