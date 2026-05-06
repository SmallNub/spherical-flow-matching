import torch
import torch.utils.data as data
from rvf.manifolds.sphere import SphereManifold
from rvf.manifolds.hyperboloid import HyperboloidManifold
import matplotlib.pyplot as plt


class EmbeddingDataset(data.Dataset):
    def __init__(self, path):
        self.data_dict = torch.load(path, weights_only=True)
        self.data = self.data_dict["encodings"]
        self.splits = self.data_dict["split_ids"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]