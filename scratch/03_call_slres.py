import polars as pl
import slres
from pprint import pprint
from alive_progress import alive_bar
import os

skip_existing = True
out_file = 'proc/residuals.parquet'

df = pl.read_parquet("proc/passes_joined.parquet").sort(
    "file_path", "station_id", "pass_number_in_file"
)

df_lens = (
    df.group_by("file_path", "station_id")
    .agg(pl.len().alias("total_passes"))
    .sort("file_path", "station_id")
)

total_passes = df_lens['total_passes'].sum()

with alive_bar(total_passes) as bar:
    for d in df_lens.iter_rows(named=True):
        bar.title(os.path.split(d["file_path"])[-1])
        df_this = df.filter(
            pl.col("file_path") == d["file_path"],
            pl.col("station_id") == d["station_id"],
        )

        for dd in df_this.iter_rows(named=True):
            try:
                print(dd)

                output_path = os.path.join(
                    out_file,
                    f"target_name={dd['target_name']}",
                    f"year={dd['year']}",
                    f"file_name={dd['file_name']}",
                    f"pass_number_in_file={dd['pass_number_in_file']}",
                    f"00000000.parquet",
                )
                if os.path.exists(output_path) and skip_existing:
                    print(f"Skipping {dd['file_name']}...")
                    bar(skipped=True)
                    continue

                _df = slres.process_one(
                    dd["file_path"],
                    dd["file_path_right"],
                    dd["station_id"],
                    dd["pass_number_in_file"],
                    wavelength=dd["transmit_wave"],
                    out_dir="out",
                    verbose=True,
                )
                _df = _df.with_columns(
                    *[pl.Series(name=k, values=_df.height * [v]) for k,v in dd.items() if k not in ['datetime_start', 'datetime_end', 'datetime_mid', 'pass_end_bound', 'obs']]
                )
                _df.write_parquet(out_file, partition_by=['target_name', 'year', 'file_name', 'pass_number_in_file'])
                print(_df.height)
                bar()
            except RuntimeError:
                print("RuntimeError occurred")
