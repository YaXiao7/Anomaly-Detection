from autoquery.queries import Query
from autoquery.scenarios import query_and_preserve
from threading import Thread
from random import randint
import datetime
import time
import subprocess
import os
from fault import Fault, inject_fault

url = "http://10.170.69.50:3002"
# how long to collect data, in minitues
DURATION = 120


def query_func():
    q = Query(url)
    q.login()
    query_and_preserve(q)


def flask_app():
    from flask import Flask

    app = Flask(__name__)
    app.debug = False

    @app.route("/fault")
    def fault():
        return str(injected_fault.value) + " " + str(injected_timestamp)

    app.run(host="0.0.0.0", port=3000)


start_time = datetime.datetime.now()
all_faults = [
    Fault.CPU,
    Fault.MEMORY,
    Fault.NETWORK_LOSS,
    Fault.NETWORK_LATENCY,
]
fault_str = {
    Fault.CPU: "cpu",
    Fault.MEMORY: "memory",
    Fault.NETWORK_LOSS: "loss",
    Fault.NETWORK_LATENCY: "latency",
}
fault_type = {
    Fault.CPU: "StressChaos",
    Fault.MEMORY: "StressChaos",
    Fault.NETWORK_LOSS: "NetworkChaos",
    Fault.NETWORK_LATENCY: "NetworkChaos",
}
remain_faults = all_faults[:]

injected_fault: Fault = Fault.CPU
injected_timestamp = 0.0

if __name__ == "__main__":
    # tt = Thread(target=flask_app)
    # tt.start()

    while True:
        this_hour_now = datetime.datetime.now()
        if (this_hour_now - start_time).total_seconds() > DURATION * 60:
            break

        # inject 10 minutes fault
        r = 0
        if len(remain_faults) == 0:
            remain_faults = all_faults[:]
        if len(remain_faults) == 1:
            r = 0
        else:
            r = randint(0, len(remain_faults) - 1)
        injected_fault = remain_faults[r]
        injected_timestamp = int(this_hour_now.timestamp())
        inject_fault(injected_fault, injected_timestamp)
        remain_faults.remove(injected_fault)
        print(f"inject fault: {injected_fault}")

        time.sleep(5)

        # write fault to file
        with open("fault.txt", "a") as f:
            f.write(str(injected_fault) + " " + str(injected_timestamp) + "\n")
            result = subprocess.run(
                f"kubectl get {fault_type[injected_fault]} -n trainticket {fault_str[injected_fault]}-{injected_timestamp} -o jsonpath='{{.status.experiment.containerRecords[:].id}}'",
                stdout=subprocess.PIPE,
                shell=True,
            )
            s = result.stdout.decode("utf-8")
            s = s.split(" ")
            s = list(map(lambda x: x.split("/")[1], s))
            for ss in s:
                f.write(ss + "\n")

        while True:
            now = datetime.datetime.now()
            # if (now - this_hour_now).total_seconds() > 10 * 60:
            #     print("fault recovered")
            #     current_fault = Fault.NONE
            if (now - this_hour_now).total_seconds() > 20 * 60:
                break

            stress = randint(10, 20)
            print(f"stress: {stress}")
            tasks: list[Thread] = []
            for i in range(stress):
                t = Thread(target=query_func)
                t.start()
                tasks.append(t)
            for thread in tasks:
                thread.join()
