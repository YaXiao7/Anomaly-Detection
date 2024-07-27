import argparse
import os
import torch
import random
import time
from enum import Enum

from exp.exp_informer import Exp_Informer
from packages.Client.FedClient import FedClient

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

parser.add_argument(
    "--clientname", type=str, required=True, help="fed client name, used for server to get the weight of this client"
)
parser.add_argument("--dataname", type=str, required=True, help="custom data file name")

args = parser.parse_args()  # 为模型填写参数

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(" ", "")
    device_ids = args.devices.split(",")
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    "ETTh1": {"data": "ETTh1.csv", "T": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
    "ETTh2": {"data": "ETTh2.csv", "T": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
    "ETTm1": {"data": "ETTm1.csv", "T": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
    "ETTm2": {"data": "ETTm2.csv", "T": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
    "test": {"data": "test.csv", "T": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
    "custom": {"data": args.dataname, "T": "fault", "M": [6, 6, 6], "S": [1, 1, 1], "MS": [6, 6, 2]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info["data"]
    args.target = data_info["T"]
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(" ", "").split(",")]
args.detail_freq = args.freq
args.freq = args.freq[-1:]


class EResult(Enum):
    REAL_TRUE_PRED_TRUE = 0
    REAL_TRUE_PRED_FALSE = 1
    REAL_FALSE_PRED_TRUE = 2
    REAL_FALSE_PRED_FALSE = 3


def process_result(preds, real) -> EResult:
    preds = preds[0].reshape(-1)
    real = real[0].reshape(-1)
    if real[0] == 1:
        # real true
        if preds[1] >= 0.5:
            # pred true
            return EResult.REAL_TRUE_PRED_TRUE
        else:
            # pred false
            return EResult.REAL_TRUE_PRED_FALSE
    elif real[0] == 0:
        # real false
        if preds[1] >= 0.5:
            # pred true
            return EResult.REAL_FALSE_PRED_TRUE
        else:
            # pred false
            return EResult.REAL_FALSE_PRED_FALSE
    else:
        raise Exception(f"invalid value for real[i]: {real[0]}")


exp = Exp_Informer(args)
model = exp.model
fedclient = FedClient(args.clientname)
fedclient.set_server("localhost", 5001)

real_true_pred_true = 0
real_true_pred_false = 0
real_false_pred_true = 0
real_false_pred_false = 0

SPLIT_LINE = "=" * 20


def print_result():
    print(f"real_true_pred_true: {real_true_pred_true}")
    print(f"real_true_pred_false: {real_true_pred_false}")
    print(f"real_false_pred_true: {real_false_pred_true}")
    print(f"real_false_pred_false: {real_false_pred_false}")
    print(SPLIT_LINE)


def get_end():
    return random.randint(300, 1300)


trained_times = 1
while True:
    latest_version = fedclient.request_model_version()
    params = fedclient.request_model_params()
    if latest_version > fedclient.current_model_version:
        new_params = exp.train(params)
        fedclient.update(new_params)
        fedclient.current_model_version = latest_version
        trained_times += 1

    # use model to predict
    preds, real = exp.predict(params, get_end())
    r = process_result(preds, real)
    if r == EResult.REAL_TRUE_PRED_FALSE:
        real_true_pred_true += 1
    elif r == EResult.REAL_TRUE_PRED_FALSE:
        real_true_pred_false += 1
    elif r == EResult.REAL_FALSE_PRED_TRUE:
        real_false_pred_true += 1
    elif r == EResult.REAL_FALSE_PRED_FALSE:
        real_false_pred_false += 1

    print_result()
    time.sleep(10)
    # trained_times += 1
