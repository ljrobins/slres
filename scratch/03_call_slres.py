import polars as pl
import slres
from pprint import pprint
from alive_progress import alive_bar
import os

df = pl.read_parquet("proc/passes_joined.parquet").sort(
    "file_path", "station_id", "pass_start_line"
)

df_lens = df.group_by("file_path", "station_id").agg(pl.len().alias("total_passes")).sort('file_path', 'station_id')

with alive_bar(df_lens.height) as bar:
    for d in df_lens.iter_rows(named=True):
        bar.title(os.path.split(d['file_path'])[-1])
        df_this = df.filter(
            pl.col("file_path") == d["file_path"], pl.col("station_id") == d["station_id"]
        ).sort("pass_start_line")

        for i, dd in enumerate(df_this.iter_rows(named=True)):
            try:
                print(i, dd)
                slres.process_one(
                    dd["file_path"],
                    dd["file_path_right"],
                    dd["station_id"],
                    i,
                    wavelength=dd["transmit_wave"],
                    out_dir="out",
                    verbose=True,
                )
            except RuntimeWarning:
                pass
        bar()