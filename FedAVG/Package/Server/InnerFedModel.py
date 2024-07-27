from torch.nn import Module
from torch import Tensor, tensor
import logging
import requests
import pickle

logging.basicConfig(level=logging.INFO)


class ReceivedParam:
    previous_model_version: int
    client: str
    model_pram: dict[str, list]

    def __init__(self, previous_model_version, client, model_param):
        self.previous_model_version = previous_model_version
        self.client = client
        self.model_param = model_param


class InnerFedModel:
    _model: Module
    _received_params: list[ReceivedParam]
    _current_model_version: int

    def __init__(self, model: Module) -> None:
        self._model = model
        self._received_params = []
        self._current_model_version = 0

    def add_param(self, data: ReceivedParam):
        self._received_params.append(data)

    def get_matrix(self, ip, port) -> dict[str, int]:
        r = requests.get(f"http://{ip}:{port}/matrix")
        r = r.content
        mat = pickle.loads(r)
        return mat

    def update_model(self):
        logging.info("begin to agg")

        params_len = len(self._received_params)
        if params_len == 0:
            logging.warning("no params received")
            return

        mat = self.get_matrix("127.0.0.1", "5000")

        t = self._received_params.pop(0)
        while t.client not in mat.keys():
            t = self._received_params.pop(0)

        new_param = {}
        for k, v in t.model_param.items():
            new_param[k] = tensor(v) * mat[t.client]

        for param in self._received_params:
            if param.client not in mat.keys():
                continue
            for k, v in param.model_param.items():
                new_param[k] += tensor(v) * mat[param.client]

        self._model.load_state_dict(new_param)
        self._current_model_version += 1

        self._received_params = []
