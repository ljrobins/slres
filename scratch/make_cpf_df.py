import polars as pl
import os
import datetime

in_dir = "/media/liam/Backup/data/slr/data/cpf_predicts/"

data = []
for root, dirs, files in os.walk(in_dir):
    print(root)
    for file in files:
        if file.endswith(".hts"):
            data.append({})
            file_path = os.path.join(root, file)
            with open(file_path) as f:
                l1 = f.readline().split()
                l2 = f.readline().split()
                data[-1]["file_path"] = file_path
                data[-1]["target_name"] = l1[-2]
                data[-1]["datetime_start"] = datetime.datetime(
                    *[int(x) for x in l2[4:10]]
                )  # UTC
                data[-1]["datetime_end"] = datetime.datetime(
                    *[int(x) for x in l2[10:16]]
                )
pl.DataFrame(data).write_parquet("proc/cpf_metadata.parquet")
