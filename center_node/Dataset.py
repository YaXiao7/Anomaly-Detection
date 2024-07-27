from torch.utils.data import Dataset
from utils import name2idx
import os
import torch
import pickle


class MyDataset(Dataset):
    len: int
    data: list[tuple[torch.Tensor, torch.Tensor]]

    def __init__(self):
        dirs = os.listdir("./embedding/")
        self.len = len(dirs)

        self.data = []
        for _ in range(self.len):
            self.data.append((torch.Tensor(), torch.Tensor()))

        for dir in dirs:
            x = torch.zeros([11, 1024])
            y = torch.zeros(11, dtype=torch.long)
            services = os.listdir("./embedding/" + dir)
            for service in services:
                with open("./embedding/" + dir + "/" + service, "rb") as f:
                    pkl_content = pickle.load(f)
                    x[name2idx(service)] = torch.tensor(pkl_content["embedding"])
                    y[name2idx(service)] = torch.tensor(1) if pkl_content["isFaultService"] else torch.tensor(0)

            self.data[int(dir)] = (x, y)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def to(self, device: str):
        self.data = [(x.to(device), y.to(device)) for x, y in self.data]
        return self
