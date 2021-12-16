from Trajectory import Trajectory


class Sample:
    """
    This class makes the traj of a particular id and then makes a lot of samples of traj of stride=1
    """
    def __init__(self, id, dataset) -> None:
        self.traj = Trajectory(id, dataset)
        self.seq_len = 20
        self.min_len = 10
        self.samples = self.slice()
    
    def __len__(self):
        return len(self.traj)

    def slice(self):

        if len(self) < self.min_len:
            return []

        split_samples = []
        start_idx = 0
        end_idx = self.seq_len
        
        while start_idx < len(self):
            new_traj = self.traj.positions[start_idx : start_idx+self.seq_len]
            start_idx += 1 
        
            if len(new_traj) < self.min_len:
                continue
            split_samples.append(new_traj)
        
        return split_samples
    