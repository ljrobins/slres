import polars as pl
import os
from datetime import date
from alive_progress import alive_bar
import humanize
import uuid

import os


def get_size(start_path: str):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def process_one(fpath: str) -> pl.DataFrame:
    file_name = os.path.split(fpath)[-1]
    laser_config = {}
    data = []
    with open(fpath, "r") as f:
        lines = list(f)
        df_line_and_id = (
            pl.DataFrame(
                [
                    {"line_num": i, "station_id": int(l.split()[2])}
                    for i, l in enumerate(lines)
                    if l.lower().startswith("h2")
                ]
            )
            .sort("station_id", "line_num")
            .with_columns(
                (pl.cum_count("station_id").over("station_id").alias("pass_index") - 1)
            )
        )

        for i, line in enumerate(lines):
            line = line.lower()
            ls = line.split()
            if line.startswith("h1"):  # Just to get file information
                header_done = False
                file_info = {
                    "file_name": file_name,
                    "file_path": fpath,
                    "file_year": file_name.split("_")[-1][:4],
                }
            if line.startswith("h2"):  # sec 1.2, pg. 4, Station Header
                station_header = {}
                station_header["station_name"] = ls[1]
                station_header["station_id"] = int(ls[2])
                station_header["station_time_scale"] = int(ls[5])
                station_header["pass_number_in_file"] = df_line_and_id.filter(
                    pl.col("station_id") == station_header["station_id"],
                    pl.col("line_num") == i,
                )["pass_index"][0]
            if line.startswith("h3"):  # sec 1.3, pg. 5, Target Header
                target_header = {}
                target_header["target_name"] = fpath.split("/")[-1].split("_")[0]
                # target_header["ilrs_sat_id"] = ls[
                #     2
                # ] # Based on COSPAR ID, says the doc
                target_header["sc_id_code"] = int(ls[3])
                # target_header["norad_id"] = int(ls[4])
                target_header["sc_epoch_time_scale"] = int(ls[5])
                target_header["target_type"] = int(ls[6]) if len(ls) > 6 else None
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
            if line.startswith("c0"):  # sec 2.1, pg. 10, System Configuration Record
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
                if header_done:
                    continue
                d = {}
                header_done = True
                if (
                    laser_config
                    and system_config
                    and pass_header
                    and target_header
                    and file_info
                    and station_header
                ):
                    # Just saving one record for each
                    ls = line.split()
                    d["sec"] = float(ls[1])
                    d["ltt"] = float(ls[2])
                    d["config"] = ls[3]
                    d["filter_flag"] = ls[4]

                    d.update(
                        **laser_config,
                        **system_config,
                        **pass_header,
                        **target_header,
                        **file_info,
                        **station_header,
                    )
                else:
                    print(f"Skipping some data in {fpath} due to missing headers...")
                for k, v in d.items():
                    if isinstance(v, (float, int)):
                        if v < 0:
                            d[k] = None
                data.append(d)

    overrides = {
        pl.UInt32: ["norad_id"],
        pl.UInt16: [
            "sc_id_code",
        ],
        pl.Float32: [
            "pulse_fwhm_ps",
            "beam_div_arcsec",
            "primary_wave",
            "transmit_wave",
            "nom_fire_rate_hz",
            "pulse_energy_mj",
        ],
        pl.UInt8: [
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


in_dir = "/media/liam/Backup/data/slr/data/fr_crd/"
out_file = os.path.join("proc", "frd_passes.parquet")
skip_existing = True

total_bytes_to_scan = get_size(in_dir)
completed_bytes = 0

with alive_bar(total_bytes_to_scan, unit="B") as bar:
    for root, dirs, files in os.walk(in_dir, topdown=False):
        for file in files:
            add_to_bad = False
            ifile = os.path.join(root, file)

            # print(f'Scanned {humanize.naturalsize(completed_bytes)}/{humanize.naturalsize(total_bytes_to_scan)}')
            with open("slres/badfiles.txt", "r") as f:
                bad_files = [x.strip() for x in f.readlines()]

            if not file.endswith(".frd") or file in bad_files:
                print(f"Skipping because {file} is bad or not frd...")
                bar(os.path.getsize(ifile), skipped=True)
                continue

            file_year = file.split("_")[-1][:4]
            target_name = file.split("/")[-1].split("_")[0]
            output_path = os.path.join(
                out_file,
                f"file_year={file_year}",
                f"target_name={target_name}",
                f"file_name={file}",
                f"00000000.parquet",
            )
            if os.path.exists(output_path) and skip_existing:
                print(f"Skipping {ifile}...")
                bar(os.path.getsize(ifile), skipped=True)
                continue
            print(f"Processing {ifile}")
            try:
                df = process_one(ifile)
                if df.width > 0:
                    df.write_parquet(
                        out_file, partition_by=["file_year", "target_name", "file_name"]
                    )
                    bar(os.path.getsize(ifile))
                else:
                    add_to_bad = True

            except UnicodeDecodeError as e:
                print(f"{e} on {ifile}, adding to bad files")
                add_to_bad = True

            if add_to_bad:
                print(f"File {file} was bad in some way, adding it to the bad list")
                with open("slres/badfiles.txt", "w") as f:
                    bad_files.append(file)
                    f.writelines("\n".join(bad_files))
                bar(os.path.getsize(ifile), skipped=True)
