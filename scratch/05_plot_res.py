import polars as pl
import matplotlib.pyplot as plt
import numpy as np

df = pl.read_parquet('proc/residuals.parquet')

for d in sorted(df.partition_by('pass_uid'), key=lambda x: x.height, reverse=True):
    plt.scatter(df['seconds_of_day'], df['residual'])
    plt.savefig('test.png')
endd

res = df['residual'].drop_nans().to_numpy()
plt.hist(res, bins=1000, range=np.percentile(res, [0.1, 99.9]))
plt.savefig('test.png')