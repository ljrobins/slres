import polars as pl

df_frd = pl.scan_parquet("proc/slr_passes.parquet")

df_pass = (
    df_frd.group_by(
        ["file_name", "station_id", "transmit_wave", "pass_start_date", "pass_uid"]
    )
    .agg(pl.len().alias("obs"), pl.col("sec").first().alias("pass_start_seconds"))
    .with_columns(
        (
            pl.col("pass_start_date").cast(pl.Datetime)
            + pl.duration(seconds=pl.col("pass_start_seconds"))
        ).alias("pass_start_datetime")
    )
    .collect(engine="streaming")
)

print(df_pass)

df_cpf = pl.read_parquet("proc/cpf_metadata.parquet")

print(df_cpf)

df_passes = df_pass.join_where(
    df_cpf,
    pl.col("datetime_start") < pl.col("pass_start_datetime"),
    pl.col("datetime_end") > pl.col("pass_start_datetime"),
    pl.col('target_name') == pl.col('target_name_right'),
)

print(df_passes)
