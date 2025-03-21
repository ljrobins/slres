import datetime as dt
from typing import Tuple
import os
import polars as pl

src_dir = './slres'

def mjd2cal(mjd: float) -> Tuple[int, int, int]:
    # Return the calendar year, month and day of month from the Modified Julian Day

    mil = 51543.0
    milm = dt.datetime(1999, 12, 31, 0, 0, 0)
    deld = mjd - mil
    ret = milm + dt.timedelta(days=deld)

    return ret.year, ret.month, ret.day

def snx_coords(mjd: float) -> pl.DataFrame:
    a = mjd2cal(mjd)
    b = dt.datetime(a[0], a[1], a[2])
    year = b.timetuple().tm_yday
    PVid = open(os.path.join(src_dir, "SLR_PV.snx"), "r")
    PVsite = False
    PVxyz = False
    data = {}

    for line in PVid:
        if "-SITE/ID" in line:
            PVsite = False
        if PVsite:
            code = line[1:5]
            if code not in data:
                data[code] = {}
                data[code]['STAT_id'] = code

            data[code]['lon'] = (
                float(line[44:47])
                + float(line[48:50]) / 60.0
                + float(line[51:55]) / 3600.0
            )
            data[code]['lat'] = (
                float(line[56:59])
                + float(line[59:62]) / 60.0
                + float(line[63:67]) / 3600.0
            )
            data[code]['hei'] = float(line[68:75])
            data[code]['STAT_name'] = line[21:32].strip()
        if "+SITE/ID" in line:
            PVsite = True
            PVid.readline()

        if "-SOLUTION/ESTIMATE" in line:
            PVxyz = False
            break
        if PVxyz:
            code = line[14:18]
            refE = line[27:39].split(":")
            tnow = dt.datetime.now()
            year = int(refE[0])
            if tnow.year - 2000 < year:
                year = year + 1900
            else:
                year = year + 2000
            PVdt = (
                dt.datetime(year, 1, 1)
                + dt.timedelta(days=int(refE[1]) - 1)
                + dt.timedelta(seconds=int(refE[2]))
            )
            STAT_X = float(line[47:68])
            line = PVid.readline()
            STAT_Y = float(line[47:68])
            line = PVid.readline()
            STAT_Z = float(line[47:68])
            line = PVid.readline()
            VEL_X = float(line[47:68])
            line = PVid.readline()
            VEL_Y = float(line[47:68])
            line = PVid.readline()
            VEL_Z = float(line[47:68])
        
            a = mjd2cal(mjd)
            tdt = dt.datetime(a[0], a[1], a[2]) + dt.timedelta(seconds=86400.0 * (mjd % 1))

            delt = (tdt - PVdt).total_seconds() / (365.25 * 86400.0)

            STAT_X = STAT_X + delt * VEL_X
            STAT_Y = STAT_Y + delt * VEL_Y
            STAT_Z = STAT_Z + delt * VEL_Z

            data[code]['STAT_X'] = STAT_X * 1e-6
            data[code]['STAT_Y'] = STAT_Y * 1e-6
            data[code]['STAT_Z'] = STAT_Z * 1e-6

        if "+SOLUTION/ESTIMATE" in line:
            PVxyz = True
            PVid.readline()
    PVid.close()

    return pl.DataFrame(list(data.values()))

print(snx_coords(100))
