from flask import Flask, request
from threading import Lock
import logging
from queue import Queue
from pickle import loads


logging.basicConfig(level=logging.INFO)


def run_server(signal: Queue, m_model: Lock, ip, port):
    logging.info("into run server")

    output_message = Queue(1)

    app = Flask(__name__)

    @app.route("/update", methods=["POST"])
    def update():
        # data structure:
        # {
        #   "previous_model_version": int
        #   "clieant": string
        #   "model_param": dict[str, list]
        # }
        data = loads(request.data)
        logging.info(f"receive update param from {data['client']}")
        m_model.acquire()
        signal.put({"type": "get_version", "queue": output_message})
        if data["previous_model_version"] != output_message.get():
            return {"status": "incorrect model version"}
        signal.put({"type": "add_param", "data": data["model_param"]})
        m_model.release()
        return {"status": "success"}
    
    @app.route("/version", methods=["GET"])
    def version():
        return {"status": "success"}
    
    @app.route("/model", methods=["POST"])
    def model():
        return {"status": "success"}

    app.run(ip, port)
