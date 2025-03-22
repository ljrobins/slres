import polars as pl
import os
from datetime import date
from alive_progress import alive_bar
import humanize
import uuid


def process_one(fpath: str) -> pl.DataFrame:
    file_name = os.path.split(fpath)[-1]
    laser_config = {}
    data = []
    with open(fpath, "r") as f:
        lines = list(f)
        with alive_bar(len(lines), title=os.path.split(fpath)[-1]) as bar:
            for line in lines:
                bar()
                line = line.lower()
                ls = line.split()
                if line.startswith("h1"):  # Just to get file information
                    file_info = {"file_name": file_name}
                if line.startswith("h2"):  # sec 1.2, pg. 4, Station Header
                    station_header = {}
                    station_header["station_name"] = ls[1]
                    station_header["station_id"] = int(ls[2])
                    station_header["station_time_scale"] = int(ls[5])
                if line.startswith("h3"):  # sec 1.3, pg. 5, Target Header
                    target_header = {}
                    target_header["target_name"] = ls[1]
                    # target_header["ilrs_sat_id"] = ls[
                    #     2
                    # ] # Based on COSPAR ID, says the doc
                    target_header["sc_id_code"] = int(ls[3])
                    target_header["norad_id"] = int(ls[4])
                    target_header["sc_epoch_time_scale"] = int(ls[5])
                    target_header["target_type"] = int(ls[6])
                if line.startswith("h4"):  # sec 1.4, pg. 6, Session (Pass) Header
                    pass_header = {}
                    assert ls[1][0] == "0", (
                        f"The Data type is not full rate? (found {ls[1]} instead of '0')"
                    )  # It should always be full rate for this script
                    pass_header["pass_start_date"] = date(
                        int(ls[2]), int(ls[3]), int(ls[4])
                    )
                    # Not parsing H/M/S or end date out of convenience
                    pass_header["data_release"] = int(ls[14])
                    pass_header["refraction_corrected"] = bool(int(ls[15]))
                    pass_header["com_corrected"] = bool(int(ls[16]))
                    pass_header["rec_amp_corrected"] = bool(int(ls[17]))
                    pass_header["system_delay_applied"] = bool(int(ls[18]))
                    pass_header["sc_delay_applied"] = bool(int(ls[19]))
                    pass_header["range_type"] = int(ls[20])
                    pass_header["data_quality"] = int(ls[21])
                    pass_header["pass_uid"] = str(uuid.uuid4())
                if line.startswith(
                    "c0"
                ):  # sec 2.1, pg. 10, System Configuration Record
                    system_config = {}
                    system_config["transmit_wave"] = float(ls[2])
                    system_config["system_config_id"] = ls[3]
                if line.startswith("c1"):  # sec 2.2, pg. 11, Laser Configuration Record
                    laser_config = {}
                    laser_config["laser_config_id"] = ls[2]
                    laser_config["laser_type"] = ls[3]
                    laser_config["primary_wave"] = float(ls[4])
                    laser_config["nom_fire_rate_hz"] = float(ls[5])
                    laser_config["pulse_energy_mj"] = float(ls[6])
                    laser_config["pulse_fwhm_ps"] = float(ls[7])
                    laser_config["beam_div_arcsec"] = float(ls[8])
                    laser_config["semi_train_pulses"] = int(ls[9])
                if line.startswith("10"):  # Data record
                    d = {}
                    ls = line.split()
                    d["sec"] = float(ls[1])
                    d["ltt"] = float(ls[2])
                    d["config"] = ls[3]
                    d["filter_flag"] = ls[4]
                    if (
                        laser_config
                        and system_config
                        and pass_header
                        and target_header
                        and file_info
                        and station_header
                    ):
                        d.update(
                            **laser_config,
                            **system_config,
                            **pass_header,
                            **target_header,
                            **file_info,
                            **station_header,
                        )
                    else:
                        print(
                            f"Skipping some data in {fpath} due to missing headers..."
                        )
                        print(
                            f"{laser_config=}, {system_config=}, {pass_header=}, {target_header=}, {file_info=}, {station_header=}"
                        )
                        endd
                        continue
                    for k, v in d.items():
                        if isinstance(v, (float, int)):
                            if v < 0:
                                d[k] = None
                    data.append(d)

    overrides = {
        pl.UInt32: ["norad_id"],
        pl.UInt16: [
            "nom_fire_rate_hz",
            "primary_wave",
            "transmit_wave",
            "pulse_energy_mj",
            "sc_id_code",
        ],
        pl.UInt8: [
            "pulse_fwhm_ps",
            "beam_div_arcsec",
            "semi_train_pulses",
            "range_type",
            "data_quality",
            "data_release",
            "target_type",
            "sc_epoch_time_scale",
            "station_time_scale",
            "filter_flag",
        ],
    }
    schema_overrides = {col: dtype for dtype, cols in overrides.items() for col in cols}
    df = pl.DataFrame(data, schema_overrides=schema_overrides)
    return df


# in_dir = "/media/liam/Backup/data/slr/data/fr_crd/technosat/2020/"
in_dir = "data"
out_file = os.path.join("proc", "slr_passes.parquet")
skip_existing = False

with open("slres/badfiles.txt", "r") as f:
    bad_files = [x.strip() for x in f.readlines()]

for root, dirs, files in os.walk(in_dir, topdown=False):
    for file in files:
        if not file.endswith(".frd"):
            continue
        ifile = os.path.join(root, file)
        output_path = os.path.join(out_file, f"file_name={file}", "00000000.parquet")
        if os.path.exists(output_path) and skip_existing:
            print(f"Skipping {ifile}...")
            continue
        if file in bad_files:
            print(f"This file is known to be bad, skipping...")
            continue
        print(f"Processing {ifile}")
        df = process_one(ifile)
        df.write_parquet(out_file, partition_by="file_name")
        print(
            f"Wrote: {humanize.naturalsize(os.path.getsize(output_path))} .parquet ({humanize.naturalsize(df.estimated_size())} in memory, original was {humanize.naturalsize(os.path.getsize(ifile))})"
        )
