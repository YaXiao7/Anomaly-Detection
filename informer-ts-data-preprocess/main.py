import argparse
import os
import pandas as pd
import os.path as path
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transfer metric data")
    parser.add_argument("-d", "--data_dir", help="all experience directory", required=True)
    args = parser.parse_args()

    outdir = "output"
    if not path.exists(outdir):
        os.mkdir(outdir)

    all_exps = ["exp1", "exp2", "exp3", "exp4", "exp5"]
    
    # {podname:poddata}
    all_pod_data:dict[str, pd.DataFrame] = {}
    
    for expname in all_exps:
        for dirpath, dirnames, filenames in os.walk(os.path.join(args.data_dir, expname)):
            this_pod_data: pd.DataFrame = pd.DataFrame()
            contain_file = False
            for filename in filenames:
                contain_file = True
                if not "proxy" in filename:
                    this_data = pd.read_csv(path.join(dirpath, filename))
                    if not "overall" in filename:
                        container_name = filename.split(":")[0]
                        this_data.rename(columns={"cpu": f"{container_name}_cpu"}, inplace=True)
                        this_data.rename(columns={"memory": f"{container_name}_memory"}, inplace=True)
                        this_data = this_data.drop(columns=["timestamp"]).drop(columns=["fault"])
                    else:
                        this_data.drop(columns=["node"], inplace=True)
                        this_data.rename(columns={"timestamp": "date"}, inplace=True)
                        this_data["date"] = this_data["date"].astype(str)
                        for index, row in this_data.iterrows():
                            this_data.at[index, "date"] = pd.to_datetime(datetime.fromtimestamp(float(row["date"])).strftime('%Y-%m-%d %H:%M:%S'))
                            this_data.at[index, "fault"] = 0 if row["fault"] == 0 else 1
                    this_pod_data = pd.concat([this_pod_data, this_data], axis=1)
            if contain_file:
                # dirpath = ["node", "exp1", "aiops-machine-2", "ts-admin-basic-info-service-57b897cd66-w4pvs"]
                expname = dirpath.split("/")[1]
                machinename = dirpath.split("/")[2]
                podname = "-".join(dirpath.split("/")[3].split("-")[0:-2])
                if not podname in all_pod_data:
                    all_pod_data[podname] = pd.DataFrame()
                all_pod_data[podname] = pd.concat([all_pod_data[podname], this_pod_data], axis=0)
                print(f"concat exp:{expname} machine:{machinename} pod:{podname}")
            
    for podname in all_pod_data:
        print(f"write pod:{podname}")
        all_pod_data[podname].fillna(0, inplace=True)
        all_pod_data[podname].to_csv(path.join(outdir, f"{podname}.csv"), index=False)
        

