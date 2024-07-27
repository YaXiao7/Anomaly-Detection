import torch
from GAT import GAT
from Dataset import MyDataset
from utils import idx2name, name2node
from tqdm import tqdm
import time

model = GAT(1024, 1024, 2, 8, 0.5).to("cuda")

adj_max = [[], []]
for i in range(10):
    j = i + 1
    adj_max[0].append(0)
    adj_max[1].append(j)
    # adj_max[0].append(j)
    # adj_max[1].append(0)
adj_max = torch.tensor(adj_max).to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss = torch.nn.CrossEntropyLoss()

# load dataset
dataset = MyDataset().to("cuda")

model.train()
for i in tqdm(range(1000)):
    optimizer.zero_grad()
    for data in dataset:
        x = data[0]
        y = data[1]

        out = model(x, adj_max, True)
        l = loss(out[0], y)
        l.backward()
        optimizer.step()

timestamp = int(time.time())
torch.save(model, f"./models/{timestamp}.pkl")
