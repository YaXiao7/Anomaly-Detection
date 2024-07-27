import torch
import requests
import logging
from pickle import dumps
from threading import Thread, Lock
from queue import Queue
from flask import Flask, request
import pickle

# from .server_func import run_server
from .InnerFedModel import InnerFedModel, ReceivedParam

logging.basicConfig(level=logging.INFO)


class Client:
    name: str

    def __init__(self, name) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Class Client: {self.name}"


class FedServer:
    _fed_model: InnerFedModel
    # siganl type:
    # #1 {type: "add_param", data: param}
    # #2 {type: "update_model", queue: Queue}
    _signal: Queue
    _m_model: Lock

    def __init__(
        self,
        model: torch.nn.Module,
    ):
        self._fed_model = InnerFedModel(model)
        self._signal = Queue()
        self._m_model = Lock()

    def run(self, host: str, port: int):
        Thread(target=self._manager).start()

        app = Flask(__name__)

        @app.route("/update", methods=["POST"])
        def update():
            # data structure:
            # {
            #   "previous_model_version": int
            #   "client": string
            #   "model_param": dict[str, list]
            # }
            data = pickle.loads(request.data)
            logging.info(f"receive update param from {data['client']}")
            if data["previous_model_version"] > self._fed_model._current_model_version:
                return {"status": "incorrect model version"}
            self._signal.put({"type": "add_param", "data": data})
            return {"status": "success"}

        @app.route("/version", methods=["GET"])
        def version():
            return {"version": self._fed_model._current_model_version}

        @app.route("/model", methods=["GET"])
        def model():
            return pickle.dumps(self._fed_model._model.state_dict())

        Thread(target=app.run, args=(host, port)).start()

    def update_model(self):
        self._signal.put(
            {
                "type": "update_model",
            }
        )

    def save_model(self, filename):
        torch.save(self._fed_model._model, filename)

    def _manager(self):
        while True:
            signal = self._signal.get()
            if signal["type"] == "add_param":
                self._fed_model.add_param(ReceivedParam(**signal["data"]))
            elif signal["type"] == "update_model":
                self._m_model.acquire()
                self._fed_model.update_model()
                self._m_model.release()
