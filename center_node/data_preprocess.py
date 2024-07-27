import json
import os
from tqdm import tqdm

traces = []

print("Loading traces...")
for root, dirs, files in os.walk("./traces"):
    for file in tqdm(files):
        with open(f"{root}/{file}", "r") as f:
            traces = traces + json.load(f)["data"]
print(len(traces))
traces = filter(lambda trace: len(trace["spans"]) > 20, traces)

if not os.path.exists("./data"):
    os.mkdir("./data")

i = 0
print("Saving traces...")
for trace in tqdm(traces):
    if not os.path.exists(f"./data/{i}"):
        os.mkdir(f"./data/{i}")

    for span in trace["spans"]:
        name = span["operationName"].split(".")[0]
        del span["references"]
        spaninfo = json.dumps(span)
        with open(f"./data/{i}/{name}.json", "a") as f:
            f.write(spaninfo)
            f.write("\r\n")

    i += 1
