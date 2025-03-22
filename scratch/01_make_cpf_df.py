import polars as pl
import os
import datetime

in_dir = "/media/liam/Backup/data/slr/data/cpf_predicts/"
skip_existing = False
out_path = "proc/cpf_metadata.parquet"

for root, dirs, files in os.walk(in_dir):
    print(root)
    data = []

    for file in files:
        file_path = os.path.join(root, file)
        year = file_path.split("/")[-3]
        target_name = file_path.split("/")[-2]
        output_path = os.path.join(out_path, f"file_name={file}", "00000000.parquet")
        if os.path.exists(output_path) and skip_existing:
            print(f"Skipping {ifile}...")
            continue

        data.append({})
        with open(file_path) as f:
            try:
                for line in f:
                    l1 = line.split()
                    if line.startswith("H1"):
                        break
                l2 = f.readline().split()
                data[-1]["file_name"] = file
                data[-1]["year"] = year
                data[-1]["target_name"] = target_name
                data[-1]["file_path"] = file_path
                data[-1]["datetime_start"] = datetime.datetime(
                    *[int(x) for x in l2[4:10]]
                )  # UTC
                data[-1]["datetime_end"] = datetime.datetime(
                    *[int(x) for x in l2[10:16]]
                )
            except Exception as e:
                print(l1)
                print(l2)
                print(file_path)
                raise e
    if len(data):
        pl.DataFrame(data).write_parquet(out_path, partition_by=["year", "target_name"])
