from GAT import GAT
import torch
from Dataset import MyDataset
from utils import idx2name, name2node

model: GAT = torch.load("./models/1718549583.pkl").to("cuda")
dataset = MyDataset().to("cuda")
adj_mat = [[], []]
for i in range(10):
    j = i + 1
    adj_mat[0].append(0)
    adj_mat[1].append(j)
    # adj_max[0].append(j)
    # adj_max[1].append(0)
adj_mat = torch.tensor(adj_mat).to("cuda")

model.eval()

d = {}
with torch.no_grad():
    x = dataset[50][0]
    out = model(x, adj_mat, True)
    i = 0
    for caller in out[1][0][0]:
        caller_node = name2node(idx2name(caller.item()))
        callee_node = name2node(idx2name(out[1][0][1][i].item()))
        if caller_node != callee_node:
            d[f"{callee_node}"] = out[1][1][i].item()
        # print(caller_node + " -> " + callee_node + ": " + str(out[1][1][i].item()))
        i += 1

    total = 0
    for k, v in d.items():
        total += v

    for k, v in d.items():
        d[k] = v / total

print(d)

from flask import Flask
import pickle

app = Flask(__name__)


@app.route("/matrix", methods=["GET"])
def matrix():
    dd = pickle.dumps(d)
    return dd


app.run("localhost", 5000)
