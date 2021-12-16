from Dataset import Dataset
import numpy as np

class Trajectory(Dataset):
    """
    This class gives us the traj of a particular id from the dataset.
    """
    def __init__(self, id, dataset) -> None:
        self.id = id
        self.positions = dataset.full_dataset[np.where(dataset.full_dataset[:, 1] == self.id)[0]][:, 3:5]

    def __len__(self):
        return len(self.positions)
    
    def add_positions(self, positions):
        self.positions = positions