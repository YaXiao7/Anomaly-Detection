import logging
import time
import argparse
import torch
import datetime

from Package.Server.FedServer import FedServer
from models.model import Informer


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="[Informer] Long Sequences Forecasting")  # 存储参数
parser.add_argument(
    "--model",
    type=str,
    required=True,
    default="informer",
    help="model of experiment, options: [informer, informerstack, informerlight(TBD)]",
)
parser.add_argument("--data", type=str, required=True, default="ETTh1", help="data")
parser.add_argument("--root_path", type=str, default="./data/ETT/", help="root path of the data file")
parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
parser.add_argument(
    "--features",
    type=str,
    default="MS",
    help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
)
parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
)
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")

parser.add_argument("--seq_len", type=int, default=96, help="input sequence length of Informer encoder")
parser.add_argument("--label_len", type=int, default=48, help="start token length of Informer decoder")
parser.add_argument("--pred_len", type=int, default=24, help="prediction sequence length")
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
parser.add_argument("--c_out", type=int, default=7, help="output size")
parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
parser.add_argument("--s_layers", type=str, default="3,2,1", help="num of stack encoder layers")
parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
parser.add_argument("--factor", type=int, default=5, help="probsparse attn factor")
parser.add_argument("--padding", type=int, default=0, help="padding type")
parser.add_argument(
    "--distil",
    action="store_false",
    help="whether to use distilling in encoder, using this argument means not using distilling",
    default=True,
)
parser.add_argument("--CSP", action="store_true", help="whether to use CSPAttention, default=False", default=False)
parser.add_argument(
    "--dilated",
    action="store_true",
    help="whether to use dilated causal convolution in encoder, default=False",
    default=False,
)
parser.add_argument(
    "--passthrough",
    action="store_true",
    help="whether to use passthrough mechanism in encoder, default=False",
    default=False,
)
parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
parser.add_argument("--attn", type=str, default="prob", help="attention used in encoder, options:[prob, full, log]")
parser.add_argument(
    "--embed", type=str, default="timeF", help="time features encoding, options:[timeF, fixed, learned]"
)
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument("--output_attention", action="store_true", help="whether to output attention in encoder")
parser.add_argument("--do_predict", action="store_true", help="whether to predict unseen future data")
parser.add_argument("--mix", action="store_false", help="use mix attention in generative decoder", default=True)
parser.add_argument("--cols", type=str, nargs="+", help="certain cols from the data files as the input features")
parser.add_argument("--num_workers", type=int, default=0, help="data loader num workers")
parser.add_argument("--itr", type=int, default=2, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=6, help="train epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch size of train input data")
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="mse", help="loss function")
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)
parser.add_argument("--inverse", action="store_true", help="inverse output data", default=False)

parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")

parser.add_argument("--host", type=str, default="127.0.0.1", help="server listening host")
parser.add_argument("--port", type=int, default=5001, help="server listening port")
args = parser.parse_args()

if __name__ == "__main__":
    model = Informer(
        args.enc_in,
        args.dec_in,
        args.c_out,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.factor,
        args.d_model,
        args.n_heads,
        args.e_layers,  # self.args.e_layers,
        args.d_layers,
        args.d_ff,
        args.dropout,
        args.attn,
        args.embed,
        args.freq,
        args.activation,
        args.output_attention,
        args.distil,
        args.CSP,
        args.dilated,
        args.passthrough,
        torch.device("cuda"),
    ).float()
    fed_server = FedServer(model)
    fed_server.run("localhost", 5001)

    input("press enter to start")
    n_rounds = 100
    for round in range(n_rounds):
        logging.info(f"begin round {round}")
        # wait 1 minute for clients uploading new parameters
        sleep_time = 30
        logging.info(f"wait {sleep_time} seconds for new parameters")
        time.sleep(sleep_time)
        fed_server.update_model()
        logging.info(f"model version: {fed_server._fed_model._current_model_version}")
        print()
     
    now = datetime.datetime.now()    
    fed_server.save_model(f"informer_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}.model")
