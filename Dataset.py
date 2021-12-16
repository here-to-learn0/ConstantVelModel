import os
import torch
import numpy as np


class Dataset:

    def __init__(self, path) -> None:
        _, ext = os.path.splitext(path)
        if ext == ".npy":
            self.full_dataset = torch.tensor(np.load(path))
        elif ext == ".txt":
            self.full_dataset = torch.tensor(np.loadtxt(path))
        else:
            raise ValueError("Invalid extension")

    def ids(self):
        return np.unique(self.full_dataset[:, 1])
    
    def get_obj_specific_data(self, class_id: int):
        return self.full_dataset[np.where(self.full_dataset[:, 2] == class_id)[0]]
