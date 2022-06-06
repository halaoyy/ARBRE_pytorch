import numpy as np
import torch

class PointDataset(torch.utils.data.Dataset):
    """
    return format: 
        Label, UserID, ItemID
    """

    def __init__(self, dataset):
        dataset = np.array(dataset)
        self.users = dataset[:, 0]
        self.items = dataset[:, 1]
        self.labels = dataset[:, 2]

    def __getitem__(self, index):
        return [self.labels[index], self.users[index], self.items[index]]

    def __len__(self):
        return len(self.labels)


class RankDataset(torch.utils.data.Dataset):
    """
    format: 
        UserID, ItemID
    """

    def __init__(self, dataset):
        dataset = np.array(dataset)
        self.users = dataset[:, 0]
        self.items = dataset[:, 1]

    def __getitem__(self, index):
        return [self.users[index], self.items[index]]

    def __len__(self):
        return len(self.users)



