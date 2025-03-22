import polars as pl

df_frd = pl.scan_parquet("proc/frd_passes.parquet")

df_pass = (
    df_frd.group_by(
        [
            "file_path",
            "station_id",
            "transmit_wave",
            "pass_start_date",
            "pass_uid",
            "pass_number_in_file",
            "target_name",
        ]
    )
    .agg(pl.len().alias("obs"), pl.col("sec").first().alias("pass_start_seconds"))
    .with_columns(
        (
            pl.col("pass_start_date").cast(pl.Datetime)
            + pl.duration(seconds=pl.col("pass_start_seconds"))
        ).alias("pass_start_datetime"),
    )
    .with_columns(
        (pl.col("pass_start_datetime") + pl.duration(hours=1)).alias("pass_end_bound")
    )
    .collect(engine="streaming")
).sort("pass_start_datetime")

print(df_pass)

df_cpf = (
    pl.read_parquet("proc/cpf_metadata.parquet")
    .sort("datetime_start", "target_name")
    .with_columns(
        (
            pl.col("datetime_start")
            + (pl.col("datetime_end") - pl.col("datetime_start")) / 2
        ).alias("datetime_mid")
    )
    .sort("datetime_mid")
)

matching_tns = (
    df_pass.unique("target_name")
    .join(df_cpf.unique("target_name"), on="target_name")["target_name"]
    .sort()
)

djs = []
for tn in matching_tns:
    print("Target name:", tn)
    dc = df_cpf.filter(pl.col("target_name") == tn)
    print("CPF:", dc)
    df = df_pass.filter(pl.col("target_name") == tn)
    print("FRD:", df)
    dj = (
        df.lazy()
        .join_where(
            dc.lazy(),
            pl.col("datetime_start") < pl.col("pass_start_datetime"),
            pl.col("datetime_end") >= pl.col("pass_end_bound"),
        )
        .unique("pass_uid")
        .collect()
    )
    print("joined:", dj)
    djs.append(dj)
djs = pl.concat(djs).sort("file_path")

print(djs)
djs.write_parquet("proc/passes_joined.parquet")
