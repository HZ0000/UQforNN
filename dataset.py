import torch
import torch.utils.data as data

class ImdbData(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

    def __len__(self):
        return len(self.y)

def generate_data(data_dim, data_length, gt_label = False):
    x0 = torch.rand(data_length, data_dim) * 0.2
    x = torch.sin(x0)
    y = torch.sum(x, -1)
    if not gt_label:
        y += torch.normal(mean=0, std = 0.001, size = y.shape)
    return x0, y