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
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # `x` shape: (batch_size, num_tracks, input_dim)
        track_embeddings = self.set_to_set(x)  # (batch_size, num_tracks, hidden_dim)
        num_tracks = track_embeddings.shape[1]
        
        # Compute pairwise edges
        edges = []
        for i in range(num_tracks):
            for j in range(num_tracks):
                if i != j:
                    edges.append(torch.cat([track_embeddings[:, i, :], track_embeddings[:, j, :]], dim=-1))
        edges = torch.stack(edges, dim=1)  # (batch_size, num_edges, 2 * hidden_dim)
        
        edge_scores = self.edge_classifier(edges)  # (batch_size, num_edges, output_dim)
        return edge_scores
