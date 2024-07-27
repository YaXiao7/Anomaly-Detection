from enum import Enum


class Fault(Enum):
    NONE = 0
    CPU = 1
    MEMORY = 2
    NETWORK_LOSS = 3
    NETWORK_LATENCY = 4


pods = {
    "re-preserve-other": "",
    "re-preserve": "",
    "re-user": "",
    "re-food": "",
    "re-ticketinfo": "",
    "re-security": "",
    "re-contacts": "",
    "re-travel": "",
    "re-travel2": "",
    "re-station": "",
    "re-seat": "",
    "re-order-other": "",
    "re-order": "",
    "re-assurance": "",
}
# update pods information
from kubernetes import client, config

config.load_config()
v1 = client.CoreV1Api()
pos = v1.list_namespaced_pod("trainticket").items
for po in pos:
    if "ts-preserve-other-service" in po.metadata.name:
        pods["re-preserve-other"] = po.metadata.name
        continue
    elif "ts-preserve-service" in po.metadata.name:
        pods["re-preserve"] = po.metadata.name
        continue
    elif "ts-user-service" in po.metadata.name:
        pods["re-user"] = po.metadata.name
        continue
    elif "ts-food-service" in po.metadata.name:
        pods["re-food"] = po.metadata.name
        continue
    elif "ts-ticketinfo-service" in po.metadata.name:
        pods["re-ticketinfo"] = po.metadata.name
        continue
    elif "ts-security-service" in po.metadata.name:
        pods["re-security"] = po.metadata.name
        continue
    elif "ts-contacts-service" in po.metadata.name:
        pods["re-contacts"] = po.metadata.name
        continue
    elif "ts-travel-service" in po.metadata.name:
        pods["re-travel"] = po.metadata.name
        continue
    elif "ts-travel2-service" in po.metadata.name:
        pods["re-travel2"] = po.metadata.name
        continue
    elif "ts-station-service" in po.metadata.name:
        pods["re-station"] = po.metadata.name
        continue
    elif "ts-seat-service" in po.metadata.name:
        pods["re-seat"] = po.metadata.name
        continue
    elif "ts-order-other-service" in po.metadata.name:
        pods["re-order-other"] = po.metadata.name
        continue
    elif "ts-order-service" in po.metadata.name:
        pods["re-order"] = po.metadata.name
        continue
    elif "ts-assurance-service" in po.metadata.name:
        pods["re-assurance"] = po.metadata.name
        continue


def inject_fault(fault: Fault, timestamp: float):
    import subprocess
    import re

    type = ""
    content = ""
    match fault:
        case Fault.CPU:
            type = "cpu"
        case Fault.MEMORY:
            type = "memory"
        case Fault.NETWORK_LOSS:
            type = "loss"
        case Fault.NETWORK_LATENCY:
            type = "latency"
    with open(f"fault_yaml/{type}.yaml", "r") as f:
        content = f.read()
        content = re.sub("{re}", str(timestamp), content)
        content = re.sub("{re-preserve-other}", pods["re-preserve-other"], content)
        content = re.sub("{re-preserve}", pods["re-preserve"], content)
        content = re.sub("{re-user}", pods["re-user"], content)
        content = re.sub("{re-food}", pods["re-food"], content)
        content = re.sub("{re-ticketinfo}", pods["re-ticketinfo"], content)
        content = re.sub("{re-security}", pods["re-security"], content)
        content = re.sub("{re-contacts}", pods["re-contacts"], content)
        content = re.sub("{re-travel}", pods["re-travel"], content)
        content = re.sub("{re-travel2}", pods["re-travel2"], content)
        content = re.sub("{re-station}", pods["re-station"], content)
        content = re.sub("{re-seat}", pods["re-seat"], content)
        content = re.sub("{re-order-other}", pods["re-order-other"], content)
        content = re.sub("{re-order}", pods["re-order"], content)
        content = re.sub("{re-assurance}", pods["re-assurance"], content)
    subprocess.Popen(f"echo \"{content}\" | kubectl apply -f -", shell=True)
