from transformers import AutoTokenizer
import os
import pickle
from utils import isFaultService
import json

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

for path, dirs, files in os.walk("./data/"):
    example_idx = path.split("/")[2]
    for file in files:
        timestamp = 0
        file_content = ""
        service_name = file.split(".")[0]
        
        with open(os.path.join(path, file)) as f:
            file_content = f.read()
            f.seek(0)
            timestamp = json.loads(f.readline())["startTime"]
            timestamp = int(timestamp / 1000000)
            
        isFault = isFaultService(service_name, timestamp)
        pkl_content = {
            "embedding": tokenizer(file_content, max_length=1024, truncation=True)["input_ids"],
            "timestamp": timestamp,
            "isFaultService": isFault
        }
        if isFault:
            print(f"{example_idx}, {service_name}, {timestamp}, faulty")

        pkl_file = file.replace(".json", ".pkl")

        if not os.path.isdir(f"embedding/{example_idx}"):
            os.makedirs(f"embedding/{example_idx}")
        with open(f"embedding/{example_idx}/{pkl_file}", "wb") as f:
            pickle.dump(pkl_content, f)
