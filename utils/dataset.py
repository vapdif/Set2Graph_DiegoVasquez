import uproot
import numpy as np
import torch
from torch.utils.data import Dataset

class JetsDataset(Dataset):
    def __init__(self, file_path):
        self.tracks, self.labels = self._load_data(file_path)
    
    def _load_data(self, file_path):
        with uproot.open(file_path) as f:
            tracks = f["tracks"].arrays(library="np")
            labels = f["labels"].arrays(library="np")
        return np.array(tracks), np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tracks = self.tracks[idx]
        labels = self.labels[idx]
        return torch.tensor(tracks, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
