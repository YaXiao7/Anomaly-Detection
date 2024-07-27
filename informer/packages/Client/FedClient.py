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
    name: str
    server: Server
    current_model_version: int

    def __init__(self, name):
        self.name = name
        self.current_model_version = -1
        self._signal = Queue()

    def set_server(self, server_ip: str, server_port: int):
        logging.info(f"connecting to server {server_ip}:{server_port}")
        self.server = Server(server_ip, server_port)

    def request_model_version(self) -> int:
        r = requests.get(f"http://{self.server.ip}:{self.server.port}/version")
        return int(r.json()["version"])

    def request_model_params(self) -> dict[str, list]:
        r = requests.get(f"http://{self.server.ip}:{self.server.port}/model")
        return loads(r.content)

    def update(self, param: dict[str, list]):
        """
        Client updates the model to the server.
        """
        logging.info(f"update model based on version {self.current_model_version}")
        data = dumps(
            {
                "model_param": param,
                "client": self.name,
                "previous_model_version": self.current_model_version,
            }
        )
        r = requests.post(
            f"http://{self.server.ip}:{self.server.port}/update",
            data=data,
        )
        logging.info(f"update result: {r.text}")
