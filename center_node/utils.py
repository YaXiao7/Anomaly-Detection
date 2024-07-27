name_dict = {
    0: "preserve",
    1: "user",
    2: "food",
    3: "ticketinfo",
    4: "security",
    5: "contacts",
    6: "travel",
    7: "station",
    8: "seat",
    9: "order",
    10: "assurance",
}


def name2idx(name: str) -> int:
    for k, v in name_dict.items():
        if v in name:
            return k
    raise ValueError(f"Invalid name {name}")


def idx2name(idx: int) -> str:
    return name_dict[idx]


# from kubernetes import client, config

# config.load_kube_config()
# v1 = client.CoreV1Api()
# pods = v1.list_namespaced_pod("trainticket")


def name2node(name: str) -> str:
    dic = {
        "preserve": "aiops-machine-5",
        "user": "aiops-machine-a1",
        "food": "aiops-machine-5",
        "ticketinfo": "aiops-machine-a2",
        "security": "aiops-machine-5",
        "contacts": "aiops-machine-a2",
        "travel": "aiops-machine-a1",
        "station": "aiops-machine-2",
        "seat": "aiops-machine-2",
        "order": "aiops-machine-a2",
        "assurance": "aiops-machine-a2",
    }
    # for pod in pods.items:
    if name in dic:
        return dic[name]
    raise ValueError(f"Invalid name {name}")


def isFaultService(name: str, timestamp: int) -> bool:
    with open("fault.txt", "r") as f:
        while True:
            line1 = f.readline()
            fault_service = f.readline()
            if len(line1) == 0:
                break
            fault_type = line1.split(" ")[0]
            fault_timestamp = int(line1.split(" ")[1])
            if timestamp < fault_timestamp:
                break
            if timestamp <= fault_timestamp + 10 * 60:
                if name in fault_service:
                    return True
                else:
                    return False

    return False
