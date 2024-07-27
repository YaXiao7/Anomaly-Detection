import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "node/exp1/aiops-machine-2/ts-admin-basic-info-service-57b897cd66-w4pvs/ts-admin-basic-info-service:0.2.1.csv"
)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(data["timestamp"], data["cpu"], color="red")
ax2.plot(data["timestamp"], data["memory"], color="blue")

ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)

ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_visible(False)

plt.savefig("draw.jpg")
