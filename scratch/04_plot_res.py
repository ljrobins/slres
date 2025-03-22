import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet('proc/residuals.parquet')

print(df)