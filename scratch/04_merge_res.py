import polars as pl

pl.scan_parquet('proc/residuals.parquet').sink_parquet('proc/residuals_merged.parquet', engine='streaming')