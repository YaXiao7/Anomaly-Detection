import logging
import argparse
from Model import MLP
import torch
from torch.utils import data as datautils
import pandas as pd

from Package.Client.FedClient import FedClient

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="datasetname")
parser.add_argument("-e", "--epoch", type=int, default=100, help="epoch number")
parser.add_argument("-l", "--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("-b", "--batch_size", type=int, default=10, help="batch size")
parser.add_argument("-n", "--name", type=str, default="client", help="fedclient name")
parser.add_argument("-o", "--host", type=str, default="127.0.0.1", help="server listening host")
parser.add_argument("-p", "--port", type=int, default=5000, help="server listening port")
parser.add_argument("-i", "--server_host", type=str, default="127.0.0.1", help="fedserver host")
parser.add_argument("-s", "--server_port", type=int, default=5001, help="fedserver port")
args = parser.parse_args()


def datafunc(device, batch_size):
    data = pd.read_csv(args.dataset)
    x = data.iloc[:, 1:3]
    y = data.iloc[:, 3:]
    x = torch.from_numpy(x.to_numpy()).type(torch.float32).to(device)
    y = torch.from_numpy(y.to_numpy()).type(torch.float32).to(device)
    dataset = datautils.TensorDataset(x, y)
    return datautils.DataLoader(dataset, batch_size, shuffle=True)


fed_client = FedClient(
    args.name,
    MLP(input_dim=2, output_dim=1),
    args.epoch,
    args.lr,
    args.batch_size,
    args.host,
    args.port,
    data_func=datafunc,
)

fed_client.connect_server(args.server_host, args.server_port)
fed_client.run()
