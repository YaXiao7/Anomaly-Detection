import torch
from torch import nn
import requests
import logging
from queue import Queue
from threading import Thread
from flask import Flask, request
from pickle import loads, dumps


logging.basicConfig(level=logging.INFO)


class Server:
    ip: str
    port: int

    def __init__(self, ip, port) -> None:
        self.ip = ip
        self.port = port


class FedClient:
    _name: str
    _model: nn.Module
    _epoch: int
    _lr: float
    _batch_size: int
    _device: torch.device
    _server: Server
    _listening_ip: str
    _listening_port: int
    _least_model_version: int
    # singal structure:
    # #1 {type: "train", data:{model_param: param, model_version: int}, queue: Queue}
    _signal: Queue

    def __init__(self, name, model: nn.Module, epoch, lr, batch_size, ip, port, data_func):
        self._name = name
        self._epoch = epoch
        self._lr = lr
        self._batch_size = batch_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._listening_ip = ip
        self._listening_port = port
        self._least_model_version = -1
        self._signal = Queue()
        self._data_iter = data_func(self._device, self._batch_size)

    def connect_server(self, server_ip: str, server_port: int):
        logging.info(f"connecting to server {server_ip}:{server_port}")
        self._server = Server(server_ip, server_port)
        r = requests.post(
            f"http://{self._server.ip}:{self._server.port}/connect",
            json={
                "name": self._name,
                "ip": self._listening_ip,
                "port": self._listening_port,
                "port": self._listening_port,
            },
        )
        logging.info(f"connect result: {r.text}")

    def run(self):
        Thread(target=self._manager).start()

        app = Flask(__name__)

        @app.route("/train", methods=["POST"])
        def train():
            logging.info("receive train task")
            q = Queue(1)
            data = loads(request.data)
            self._signal.put(
                {
                    "type": "train",
                    "data": {
                        "model_param": data["model_param"],
                        "model_version": data["model_version"],
                    },
                    "queue": q,
                }
            )
            if q.get() == "ok":
                return {"status": "begin to train"}
            else:
                return {"status": "old model, ignore"}

        app.run(host=self._listening_ip, port=self._listening_port)

    def _manager(self):
        while True:
            signal = self._signal.get()
            if signal["type"] == "train":
                if signal["data"]["model_version"] > self._least_model_version:
                    signal["queue"].put("ok")
                    param = self._train(signal["data"]["model_param"])
                    self._least_model_version = signal["data"]["model_version"]
                    self._update(signal["data"]["model_version"], param)
                else:
                    signal["queue"].put("no")

    def _train(self, param: dict[str, any]) -> dict[str, list]:
        """
        Client trains the model on local dataset
        """
        logging.info("begin to train with initial parameters:")
        logging.info(param)

        self._load_model_with_model_param(param)
        loss = nn.MSELoss()
        trainer = torch.optim.SGD(self._model.parameters(), lr=self._lr)
        last_loss: int = 0
        for epoch in range(self._epoch):
            for x, y in self._data_iter:
                l = loss(self._model(x), y)
                trainer.zero_grad()
                l.backward()
                trainer.step()
            last_loss = loss(self._model(x), y)
        logging.info(f"epoch {epoch + 1}, loss {last_loss:f}")

        new_param: dict[str, list] = {}
        for k, v in self._model.state_dict().items():
            new_param[k] = v.tolist()
        return new_param

    def _load_model_with_model_param(self, param: dict[str, list]):
        for k, v in param.items():
            param[k] = torch.tensor(v)
        self._model.load_state_dict(param)

    def _update(self, previous_model_version: int, param: dict[str, list]):
        """
        Client updates the model to the server.
        """
        logging.info("update model with params:")
        logging.info(param)
        data = dumps(
            {
                "model_param": param,
                "client": self._name,
                "previous_model_version": previous_model_version,
            }
        )
        r = requests.post(
            f"http://{self._server.ip}:{self._server.port}/update",
            data=data,
        )
        logging.info(f"update result: {r.text}")
