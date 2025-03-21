#!/usr/bin/python3
#
# orbitNP.py
#
# Author: Matthew Wilkinson.
# Institute: Space Geodesy Facility, Herstmonceux UK.
# Research Council: British Geological Survey, Natural Environment Research Council.
#
# Version: 1.2.1
# Last Modified: 28th April 2022
#
# Visit here to read our disclaimer: http://www.bgs.ac.uk/downloads/softdisc.html and please refer to the LICENSE.txt document included in this distribution.

# Please refer to the README file included in this distribution.
#

import datetime as dt
import os
import sys
import getopt
import scipy
from scipy import interpolate
from scipy.optimize import curve_fit
from decimal import *  # noqa: F403
import numpy as np
import inspect
import polars as pl

src_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda: 0)))


def refmm(pres, temp, hum, alt, rlam, phi, hm):
    # Calculate and return refraction delay range corrections from Marini-Murray model using the pressure,
    # temperature, humidity, satellite elevation, station latitude, station height and laser wavelength

    flam = 0.9650 + 0.0164 / (rlam * rlam) + 0.228e-3 / (rlam**4)
    fphih = 1.0 - 0.26e-2 * np.cos(2.0 * phi) - 0.3 * hm
    tzc = temp - 273.15
    ez = hum * 6.11e-2 * (10.0 ** ((7.5 * tzc) / (237.3 + tzc)))
    rk = 1.163 - 0.968e-2 * np.cos(2.0 * phi) - 0.104e-2 * temp + 0.1435e-4 * pres
    a = 0.2357e-2 * pres + 0.141e-3 * ez
    b = 1.084e-8 * pres * temp * rk + (4.734e-8 * 2.0 * pres * pres) / (
        temp * (3.0 - 1.0 / rk)
    )
    sine = np.sin(alt * 2.0 * np.pi / 360.0)
    ab = a + b
    delr = (flam / fphih) * (ab / (sine + (b / ab) / (sine + 0.01)))

    tref = delr * 2.0  #  2-way delay in meters

    return tref


def dchols(a, m):
    # Perform Choleski matrix inversion

    s = np.zeros([m, m], dtype="longdouble")
    b = np.zeros([m, m], dtype="longdouble")
    x = np.zeros(m, dtype="longdouble")

    ierr = 0
    arng = np.arange(m)

    sel = np.where(a[arng, arng] <= 0.0)
    if np.size(sel) > 0:
        ierr = 2
        return ierr, a

    s[:, arng] = 1.0 / np.sqrt(a[arng, arng])
    a = a * s * s.transpose()

    for i in range(m):
        sum = a[i, i]
        if i > 0:
            for k in range(i):
                sum = sum - b[k, i] ** 2

        if sum <= 0.0:
            ierr = 3
            return ierr, a

        b[i, i] = np.sqrt(sum)

        if i != m - 1:
            for j in range(i + 1, m):
                sum = a[j, i]
                if i > 0:
                    for k in range(i):
                        sum = sum - b[k, i] * b[k, j]
                b[i, j] = sum / b[i, i]

    for i in range(m):
        for j in range(m):
            x[j] = 0.0
        x[i] = 1.0
        for j in range(m):
            sum = x[j]
            if j > 0:
                for k in range(j):
                    sum = sum - b[k, j] * x[k]
            x[j] = sum / b[j, j]

        for j in range(m):
            m1 = m - 1 - j
            sum = x[m1]
            if j > 0:
                for k in range(j):
                    m2 = m - 1 - k
                    sum = sum - b[m1, m2] * x[m2]
            x[m1] = sum / b[m1, m1]

        for j in range(m):
            a[j, i] = x[j]

    reta = a * s * s.transpose()

    return ierr, reta


def mjd2cal(mjd):
    # Return the calendar year, month and day of month from the Modified Julian Day

    mil = 51543.0
    milm = dt.datetime(1999, 12, 31, 0, 0, 0)
    deld = mjd - mil
    ret = milm + dt.timedelta(days=deld)

    return ret.year, ret.month, ret.day


def cal2mjd(yr, mm, dd):
    # Return the Modified Julian Day from the calendar year, month and day of month

    mil = 51543.0
    milm = dt.datetime(1999, 12, 31, 0, 0, 0)

    dref = dt.datetime(yr, mm, dd, 0, 0, 0)

    delt = dref - milm
    mjdref = mil + delt.days

    return mjdref


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
                data[code]["STAT_id"] = code

            data[code]["lon"] = (
                float(line[44:47])
                + float(line[48:50]) / 60.0
                + float(line[51:55]) / 3600.0
            )
            data[code]["lat"] = (
                float(line[56:59])
                + float(line[59:62]) / 60.0
                + float(line[63:67]) / 3600.0
            )
            data[code]["hei"] = float(line[68:75])
            data[code]["STAT_name"] = line[21:32].strip()
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
            tdt = dt.datetime(a[0], a[1], a[2]) + dt.timedelta(
                seconds=86400.0 * (mjd % 1)
            )

            delt = (tdt - PVdt).total_seconds() / (365.25 * 86400.0)

            STAT_X = STAT_X + delt * VEL_X
            STAT_Y = STAT_Y + delt * VEL_Y
            STAT_Z = STAT_Z + delt * VEL_Z

            data[code]["STAT_X"] = STAT_X * 1e-6
            data[code]["STAT_Y"] = STAT_Y * 1e-6
            data[code]["STAT_Z"] = STAT_Z * 1e-6

        if "+SOLUTION/ESTIMATE" in line:
            PVxyz = True
            PVid.readline()
    PVid.close()

    return pl.DataFrame(list(data.values()))


##
def SNXstats(FRdata):
    if not os.path.isfile(FRdata):
        print(f"\t+ File {FRdata} does not exist")
        sys.exit()

    PVid = open(os.path.join(src_dir, "SLR_PV.snx"), "r")
    PVsite = False
    STATnames = list()
    STATcodes = list()
    for line in PVid:
        if "-SITE/ID" in line:
            PVsite = False
        if PVsite:
            code = line[1:5]
            name = line[21:32]
            if code not in STATcodes:
                STATcodes.append(code)
                STATnames.append(name)

        if "+SITE/ID" in line:
            PVsite = True
            PVid.readline()
    PVid.close()
    sel = np.arange(len(STATcodes))

    Fcodes = list()
    if len(FRdata):
        print(f"\t+ Stations in FRD file {FRdata}")
        cmd = f"grep -e h2 -e H2 {FRdata}"
        H2 = os.popen(cmd).read().split("\n")
        for line in H2:
            a = line.split()
            if len(a) > 0:
                if (a[0] == "h2") | (a[0] == "H2"):
                    Fcodes.append(a[2])

        sel = np.where(np.isin(STATcodes, Fcodes))[0]

    ni = 0
    STATcodes = np.array(STATcodes)
    STATnames = np.array(STATnames)
    print("\n\t  #   Station Code     Station Name       Num Passes")
    for i in sel:
        nE = Fcodes.count(STATcodes[i])
        print(f"\t{ni:3d}     {STATcodes[i]}             {STATnames[i]}        {nE}")
        ni = ni + 1

    if FRdata != "":
        ist = -1
        while (ist < 0) | (ist >= ni):
            try:
                print(
                    f"\n  File contains SLR data from {ni} stations:                     (q to quit)"
                )
                rawi = input("  Select SLR station: ")
                ist = int(rawi)
            except Exception:
                if rawi == "q":
                    sys.exit()
                ist = -1

        return STATcodes[sel[ist]]
    else:
        return -1


###
# Main Program
###

# CONSTANTS
sol = 299792458.0  # Speed of Light m/s
dae = 6.378137000  # PARAMETERS OF SPHEROID
df = 298.2570

nu = 13  # lsq size
swi = 1.5e-8  # weighting applied to residual fitting
CsysID = ""  # CRD system confirguation ID
FRdata = ""  # ILRS full-rate date file in CRD format
CPFin = ""  # ILRS XYZ orbit prediction in CPF format
STAT_id = ""  # Station ID
STAT_name = ""  # Station name
STAT_abv = ""  # Satellite abbreviation
STAT_coords = ""  # Station coordinates string
SATtarget_name = ""  # Satellite name
CPFv1 = False  # Use CPF v1 predictions
SNXref = True  # Flag to acquire station coordinates fron SINEX file
SNXst = False  # Flag to acquire station coordinates fron SINEX file
METap = False  # Flag to show met data applied
Wavel = 532.0  # Laser pulse width
clipGauss = False  # GAUSS
clipLEHM = False  # Set residual clipping infront and behind the LEHM
peakDIST = False  # Peak determined from smoothed distribution. Default sets Peak as mean from a 1*sigma iteration
r50 = ""  # 50 record string
q50 = "0"  # Data quality assessment indicator
RRout = False  # Output range residuals to file
FRin = False  # Input epoch ranges in full-rate CRD
SLVout = False  # Output orbit parameters to file
Unfilter = False  # Flag to include the data in the CRD full-rate data that is flagged as 'noise'
ipass = -1  # select pass in FRD file
setupWarningList = []  # setup warnings output in summary
runWarningList = []  # run warnings output in summary
Dchannel = 0  # Detector Channel

loop = True  # loop in a FRD file
FRDloop = False  # set FRD file loop

a = scipy.__version__.split(".")
if (int(a[0]) == 0) & (int(a[1]) < 17):
    print(
        "Error: Scipy library version is too old. Version 0.17.0 or later is required"
    )
    sys.exit()

print("\n -- Check input parameters")

try:
    opts, args = getopt.getopt(
        sys.argv[1:],
        "hf:d:c:Am:p:qI:j:C:H:w:W:N:M:b:xXyz:l:g:G:k:K:u:U:Ps:Lt:onrveVFSQ",
    )
except getopt.GetoptError:
    print("Error: Check all input options")
    print("orbitNP.py -h for help option")
    sys.exit()

if np.size(opts) == 0:
    print("Error: No arguments given. Printing help options...")
    opts = [("-h", "")]

for opt, arg in opts:
    if opt in ("-f"):  # input file in full-rate CRD format
        FRdata = arg
        FRin = True
    elif opt in ("-c"):  # input satellite prediction file in CPF format
        CPFin = arg
    elif opt in ("-s"):  # input station code
        STAT_id = arg
    elif opt in ("-L"):  # list station codes
        SNXst = True
    elif opt in ("-I"):  # input system id
        CsysID = arg
    elif opt in ("-t"):  # input satellite target name
        SATtarget_name = arg
    elif opt in ("-W"):  # input laser wavelength in ns
        Wavel = float(arg)
    elif opt in ("-e"):  # set use of all CRD data points
        Unfilter = True
    elif opt in ("-V"):  # set use CPF v1
        CPFv1 = True
    elif opt in ("-g"):
        clipGauss = True
        try:
            cfactor = float(arg)
        except Exception:
            print("-g: Failed to read Gauss sigma factor")
            sys.exit()
        if (cfactor < 2.0) | (cfactor > 6.0):
            print(
                "-g: Gauss sigma rejection factor outside of permitted limits [2.0 - 6.0]"
            )
            sys.exit()
    elif opt in ("-u"):  # input upper and lower values for clipping in ps
        clipLEHM = True
        a = arg.split(":")
        if np.size(a) != 2:
            print("-u: Incorrect LEHM lower and upper input values")
            sys.exit()
        LEHMlow = np.min([float(a[0]), float(a[1])])
        LEHMupp = np.max([float(a[0]), float(a[1])])
        if LEHMlow > 0.0:
            print("\t ** Clipping not set below the LEHM")
            setupWarningList.append(
                "- Clipping not set below the LEHM, suggest '-u -"
                + str(int(LEHMlow))
                + ":"
                + str(int(LEHMupp))
                + "'"
            )
    elif opt in ("-P"):  # set Peak as high point of smotthed distribution
        peakDIST = True
    elif opt in ("-o"):  # set loop through passes in full-rate combined file
        FRDloop = True
    elif opt in ("-r"):  # set residuals output to file
        RRout = True

if SNXst:
    if STAT_id != "":
        print(f"\t+ Station code already provided as {STAT_id}")
    else:
        print("\t- List SLR Stations")
        STAT_id = SNXstats(FRdata)
        if STAT_id == -1:
            sys.exit()

if FRin:
    print("\t+ FRD file is", FRdata)
    if STAT_id == "":
        print("\t- No Station 4-digit code provided. Use -L for a list of stations")
        sys.exit()

    dataf = FRdata


df_station = snx_coords(0.0)

if STAT_id != "":
    df_stat_id = df_station.filter(pl.col("STAT_id") == STAT_id)
    assert df_stat_id.height == 1
    STAT_name = df_stat_id["STAT_name"][0]
    STAT_LAT = df_stat_id["lat"][0]

    print(f"\t+ Station is {STAT_id} {STAT_name}")

if STAT_id == "":
    print("-s: No Station ID provided")
    sys.exit()

if CsysID != "":
    print("\t+ System ID is", CsysID)
else:
    print("\t- System ID not provided")

if SATtarget_name != "":
    print("\t+ SLR Target is", SATtarget_name)

if len(CPFin):
    print("\t+ CPF Pred is", CPFin)

if Unfilter:
    print("\t+ Include unfiltered range measurements")

if CPFv1:
    print("\t+ Use CPF version 1 predictions")

elif (clipLEHM) & (clipGauss):
    print("\t- 3* Gausss sigma clipping and LEHM clipping cannot be applied together")
    sys.exit()
elif clipLEHM:
    print("\t+ Apply clipping from LEHM")
elif clipGauss:
    print("\t+ Apply n*Gauss sigma clipping")

if RRout:
    print("\t+ Output range residuals to file resids.dat")

read1st = True
while loop:  # If a multi-pass full-rate file is used, this loop will allow a quick selection of next pass.
    if not FRDloop:
        loop = False

    ep1 = -1.0  # Previous epoch
    ep1m = -1.0  # Previous epoch in met data
    mjd_daychange = False  # Change of day detectd in data records
    Mmep = list()  # Met epoch
    pressure = list()  # Pressure in mbar
    TK = list()  # Temperature in K
    humid = list()  # Humidity
    Depc = list()  # Data epochs
    Ddatet = list()  # Data datetimes
    Dmep = list()  # Data modified julian day epochs
    Drng = list()  # Data ranges
    Cmep = list()  # Calibration epoch
    Crng = list()  # Calibration system delay
    Cnum = list()  # Calibration number of points
    Crms = list()  # Calibration RMS
    Cskw = list()  # Calibration skew
    Ckurt = list()  # Calibration kurtosis
    Cpm = list()  # Calibration peak-mean
    surveyd = 0.0  # Calibration survey distance (m)
    STsel = False  # Station found in full-rate data file
    SDapplied = True  # Indictor that system delay is applied
    runWarningList = list()  # List of warnings accumulated during run
    Dchannel = -1  # Detector channel

    if FRin:
        # If the data file provided is a full-rate data file in CRD format read in the data header H4, calibrations,
        # the met data entries and the Epoch-Range data.

        print("\n -- Read FRD file for epochs, ranges and meteorological data... ")
        try:
            fid = open(FRdata)
        except IOError as e:
            print(f"\t Failed to open full-rate data file. {e.strerror} : {FRdata}")
            sys.exit()

        if read1st:  # Read through full-rate data for multiple SLR passes
            h2i = list()
            h2l = list()
            h4l = list()
            c10 = 0
            Fcount = list()
            for i, line in enumerate(fid):
                if "h2" in line or "H2" in line:
                    if c10 > 0:
                        Fcount.append(c10)
                    c10 = 0
                    h2i.append(i)
                    h2l.append(line.split()[2])
                if "h4" in line or "H4" in line:
                    h4l.append(line.strip())
                if line[0:2] == "10":
                    c10 = c10 + 1
            Fcount.append(c10)
            read1st = False

        lnpass = 1
        h2i = np.array(h2i)
        h2l = np.array(h2l)
        numpass = np.sum(h2l == STAT_id)
        selpass = np.where(h2l == STAT_id)[0]
        if numpass == 0:
            print(" EXIT: No data for station", STAT_id, "in FRD file")
            sys.exit()
        elif numpass == 1:
            print("\n FRD file contains only one pass for station", STAT_id)
            lnpass = h2i[selpass[0]]
            loop = False
            pass
        else:
            ipass = -1
            h4l = np.array(h4l)
            print(
                "\t",
                "Index",
                "\t",
                "Station Name     Num Records         H4 Start/End Entry",
            )
            for i, s in enumerate(selpass):
                print(f"\t  {i} \t  {STAT_name:11}   {Fcount[s]:8d}           {h4l[s]}")
            print(
                "\n FRD file contains",
                numpass,
                "passes for station",
                STAT_id,
                "\t\t\t(q to quit)",
            )
            while (ipass < 0) | (ipass >= numpass):
                try:
                    rawi = input("Enter pass number: ")
                    ipass = int(rawi)
                except Exception:
                    if rawi == "q":
                        sys.exit()
                    ipass = -1
            lnpass = h2i[selpass[ipass]]

        fid.seek(0)
        line = fid.readline()
        if (line.split()[0] != "H1") & (line.split()[0] != "h1"):
            print(" ERROR: FRD input file read error")
            sys.exit()

        fid.seek(0)
        h1l = ""
        for i, line in enumerate(fid):
            a = line.split()

            if (a[0] == "H1") | (a[0] == "h1"):
                CRDversion = a[2]
                h1l = line

            if (a[0] == "H2") | (a[0] == "h2"):
                if ((STAT_id == a[1]) | (STAT_id == a[2])) & (i == lnpass):
                    STsel = True
                    STAT_abv = a[1]

            if STsel:
                if (a[0] == "H3") | (a[0] == "h3"):
                    Hsat = a[1]
                    SATtarget_name = Hsat
                elif (a[0] == "H4") | (a[0] == "h4"):
                    Hyr = a[2]
                    Hmon = "{:02d}".format(int(a[3]))
                    Hd = "{:02d}".format(int(a[4]))
                    mjd1 = (
                        cal2mjd(int(a[2]), int(a[3]), int(a[4]))
                        + (float(a[5]) + float(a[6]) / 60.0 + float(a[7]) / 3600.0)
                        / 24.0
                    )
                    mjd2 = (
                        cal2mjd(int(a[8]), int(a[9]), int(a[10]))
                        + (float(a[11]) + float(a[12]) / 60.0 + float(a[13]) / 3600.0)
                        / 24.0
                    )
                    INmjd = cal2mjd(int(a[2]), int(a[3]), int(a[4]))
                    mjdm = INmjd
                    mjdc = INmjd
                    c = mjd2cal(INmjd)

                    if a[18] == "0":
                        SDapplied = False
                        print(
                            "\n -- System Delay Calibration not applied.  Will be applied"
                        )
                        runWarningList.append(
                            "System Delay Calibration was not applied to ranges"
                        )
                    else:
                        print("\n -- System Delay Calibration already applied")

                elif (a[0] == "C0") | (a[0] == "c0"):  # read from C1 entry
                    if CsysID == "":
                        CsysID = a[3]
                elif a[0] == "10":
                    if (Unfilter) | (
                        a[5] != "1"
                    ):  # Take all records or filter out the noise flags
                        ep = np.double(a[1]) / 86400.0
                        if ep1 == -1.0:
                            ep1 = ep
                        if ((ep + 300.0 / 86400.0 < mjd1 - INmjd) | (ep < ep1)) & (
                            not mjd_daychange
                        ):  # detect day change
                            print("\n -- Day change detected during pass")
                            INmjd = INmjd + 1.0
                            mjd_daychange = True
                        Dmep.append(INmjd + ep)
                        Depc.append(np.double(a[1]))
                        Drng.append(np.double(a[2]) * 1.0e12)
                        cep = np.double(a[1])

                        c = mjd2cal(INmjd)
                        datT = dt.datetime(c[0], c[1], c[2]) + dt.timedelta(seconds=cep)

                        Ddatet.append(datT)
                        if Dchannel == -1:
                            Dchannel = int(a[6])

                elif a[0] == "20":
                    epm = np.double(a[1]) / 86400.0
                    # Check if first met entry if from previous day
                    if (ep1m == -1.0) & (epm - np.mod(mjd1, 1) > 0.5):
                        print("\n -- Met dataset begins on previous day")
                        runWarningList.append("Met dataset begins on previous day")
                        mjdm = mjdm - 1.0

                    # Detect day change in met entries
                    if epm < ep1m:
                        mjdm = mjdm + 1.0

                    ep1m = epm
                    Mmep.append(mjdm + epm)
                    pressure.append(np.double(a[2]))
                    TK.append(np.double(a[3]))
                    humid.append(np.double(a[4]))
                elif a[0] == "40":
                    epc = np.double(a[1]) / 86400.0
                    Cmep.append(INmjd + epc)
                    Crng.append(np.double(a[7]))
                    Cnum.append(a[4])
                    Crms.append(a[9])
                    Cskw.append(a[10])
                    Ckurt.append(a[11])
                    Cpm.append(a[12])
                    surveyd = float(a[6])
                elif (a[0] == "H8") | (a[0] == "h8"):
                    break

        fid.close()

    Depc = np.array(Depc)
    Drng = np.array(Drng)
    Ddatet = np.array(Ddatet)
    Dmep = np.array(Dmep)
    mean_Dmep = np.mean(Dmep)  # mean data Depc

    if mjd2 < mjd1:  # Correct H4 record
        mjd2 = mjd1 + (cep.argmax() - cep.argmin()) / 86400.0

    if not SDapplied:
        if np.size(Cmep) > 1:
            IntpC = interpolate.interp1d(
                Cmep,
                Crng,
                kind="linear",
                bounds_error=False,
                fill_value=(Crng[0], Crng[-1]),
            )
            Drng = Drng - IntpC(Dmep)
        else:
            Drng = Drng - Crng[0]

    if np.size(Dmep) == 0:
        print(" No Epoch-Range data loaded, quitting...", STsel)
        sys.exit()

    if Dchannel == -1:
        Dchannel = 0

    df_stat_mjd = snx_coords(mjd1).filter(pl.col("STAT_id") == STAT_id)
    STAT_name = df_stat_mjd["STAT_name"][0]
    STAT_LAT = df_stat_mjd["lat"][0]
    STAT_LONG = df_stat_mjd["lon"][0]
    STAT_HEI = df_stat_mjd["hei"][0]
    STAT_X = df_stat_mjd["STAT_X"][0]
    STAT_Y = df_stat_mjd["STAT_Y"][0]
    STAT_Z = df_stat_mjd["STAT_Z"][0]

    if STAT_id != "":
        print("\n\t+ SLR Station is", STAT_id, STAT_name)

    print(
        f"\t+ Station Latitude, Longitude and Height: {STAT_LAT:.2f} {STAT_LONG:.2f} {STAT_HEI:.1f}"
    )

    # calculate Station lat, long coordinates in radians
    STAT_LONGrad = STAT_LONG * 2 * np.pi / 360.0
    STAT_LATrad = STAT_LAT * 2 * np.pi / 360.0
    STAT_HEI_Mm = STAT_HEI * 1e-6

    # Set up linear interpolation Python function for the met data entries
    print("\n -- Interpolate meteorological records ... ")

    if np.size(Mmep) == 0:
        METap = False
    elif np.size(Mmep) > 1:
        IntpP = interpolate.interp1d(
            Mmep,
            pressure,
            kind="linear",
            bounds_error=False,
            fill_value=(pressure[0], pressure[-1]),
        )
        IntpT = interpolate.interp1d(
            Mmep, TK, kind="linear", bounds_error=False, fill_value=(TK[0], TK[-1])
        )
        IntpH = interpolate.interp1d(
            Mmep,
            humid,
            kind="linear",
            bounds_error=False,
            fill_value=(humid[0], humid[-1]),
        )

        # Produce a met value for each data Depc using the interpolation functions
        PRESSURE = IntpP(Dmep)
        TEMP = IntpT(Dmep)
        HUM = IntpH(Dmep)

        # Test if met data epochs are suitable
        iMax = abs(np.max(Dmep) - (Mmep)).argsort()[0]
        iMin = abs(np.min(Dmep) - (Mmep)).argsort()[0]
        if (abs(np.max(Dmep) - Mmep[iMax]) < 0.5 / 24.0) & (
            abs(np.min(Dmep) - Mmep[iMin]) < 1.5 / 24.0
        ):
            METap = True
    else:
        PRESSURE = pressure[0]
        TEMP = TK[0]
        HUM = humid[0]
        METap = True

    # Read CPF Prediction File in the Depc, X, Y, Z lists and produce interpolation functions
    if len(CPFin):
        assert os.path.exists(CPFin)
        print("\n -- Read CPF prediction file:", CPFin)
        cpf_fid = open(CPFin)

        cpfEP = list()
        cpfX = list()
        cpfY = list()
        cpfZ = list()

        mep2 = 0.0
        stp = 0.0
        for line in cpf_fid:
            a = line.split()
            if a[0] == "10":
                mep = np.double(a[2]) + np.double(a[3]) / 86400.0
                if (stp == 0.0) & (mep2 != 0.0):
                    stp = mep - mep2
                mep2 = mep
                if (mep >= (mjd1 - 0.5 / 24.0) - 2.0 * stp) & (
                    mep <= (mjd2 + 0.5 / 24.0) + 3.0 * stp
                ):
                    cpfEP.append(mep)
                    cpfX.append(np.double(a[5]))
                    cpfY.append(np.double(a[6]))
                    cpfZ.append(np.double(a[7]))

        cpf_fid.close()

        cpf0 = cpfEP[0]
        if np.size(cpfEP) == 0:
            print(
                f"\n -- Selected CPF file {CPFin}does not cover the required orbit time period. Quit"
            )
            sys.exit()

        kd = 16
        if np.size(cpfEP) <= kd:
            kd = np.size(cpfEP) - 1

        # Set up linear interpolation Python function for CPF X, Y and Z components
        try:
            cpf_ply_X = np.polyfit(cpfEP - cpf0, cpfX, kd)
            cpf_ply_Y = np.polyfit(cpfEP - cpf0, cpfY, kd)
            cpf_ply_Z = np.polyfit(cpfEP - cpf0, cpfZ, kd)
        except Exception:
            kd = 9  # int(0.5*kd)+1
            if kd < 9:
                kd = len(cpfEP)
            cpf_ply_X = np.polyfit(cpfEP - cpf0, cpfX, kd)
            cpf_ply_Y = np.polyfit(cpfEP - cpf0, cpfY, kd)
            cpf_ply_Z = np.polyfit(cpfEP - cpf0, cpfZ, kd)

    mX = np.polyval(cpf_ply_X, mean_Dmep - cpf0)
    mY = np.polyval(cpf_ply_Y, mean_Dmep - cpf0)
    mZ = np.polyval(cpf_ply_Z, mean_Dmep - cpf0)
    geoR = np.sqrt(mX**2 + mY**2 + mZ**2)

    # ******

    print("\n -- Begin orbit adjustment to fit range data")
    neps = len(Depc)
    nmet = len(Mmep)

    # Calculate mid-pass time in seconds. This time is used as origin of time-dependent unknowns.
    if Depc[-1] > Depc[0]:
        dtobc = (Depc[-1] + Depc[0]) / 2.0
    else:
        dtobc = 0.0

    # generate arrays
    gvs = np.zeros(3)
    cv = np.zeros(nu)
    rhs = np.array(np.zeros(nu), order="F")
    rd = np.array(np.zeros([nu, nu]), order="F")
    rf = np.zeros(nu)
    s = np.zeros(nu)

    # zero variables
    itr = 0
    itrm = 30
    alnc = 0.0
    acrc = 0.0
    radc = 0.0
    alndc = 0.0  # Accumulated Satellite orbital time bias
    acrdc = 0.0  # Accumulated Rate of time bias
    raddc = 0.0  # Accumulated Satellite radial error
    alnddc = 0.0  # Accumulated Rate of radial error
    acrddc = 0.0  # Accumulated Acceleration of time bias
    radddc = 0.0  # Accumulated Acceleration of radial error

    alnd = 0.0
    rdld = 0.0
    alndd = 0.0
    rdldd = 0.0
    saln = 0.0
    sacr = 0.0
    srdl = 0.0

    salnd = 0.0
    sacrd = 0.0
    srdld = 0.0
    salndd = 0.0
    sacrdd = 0.0
    srdldd = 0.0

    ierr = 0

    sigt = 0.1 / 8.64e7
    sigr = 0.01 / 1.0e6
    sigtt = 0.1 / 8.64e7
    sigrr = 0.01 / 1.0e6

    oldrms = 1000.0

    ql = list()
    qm = list()
    qn = list()

    ddr = list()

    dxi = list()
    dyi = list()
    dzi = list()

    dvx = list()
    dvy = list()
    dvz = list()

    t = list()
    p = list()
    h = list()

    rX = np.polyval(cpf_ply_X, Dmep - cpf0)
    rY = np.polyval(cpf_ply_Y, Dmep - cpf0)
    rZ = np.polyval(cpf_ply_Z, Dmep - cpf0)
    cpfR = np.sqrt(
        (rX - STAT_X * 1e6) ** 2 + (rY - STAT_Y * 1e6) ** 2 + (rZ - STAT_Z * 1e6) ** 2
    )  # Range from SLR Station to satellite in metres

    if np.size(Crng) > 0:
        cpfR = cpfR + 0.5 * np.mean(Crng) * 1e-12 * sol  # Include half of system delay
    dkt = Dmep + (cpfR / sol) / 86400.0  # Time of laser light arrival at satellite
    dkt1 = dkt - 0.5 / 86400.0  # Epoch - 0.5 seconds
    dkt2 = dkt + 0.5 / 86400.0  # Epoch + 0.5 seconds

    cX = (
        np.polyval(cpf_ply_X, dkt - cpf0) * 1e-6
    )  # X component of satellite CPF prediction in megametres
    cY = (
        np.polyval(cpf_ply_Y, dkt - cpf0) * 1e-6
    )  # Y component of satellite CPF prediction in megametres
    cZ = (
        np.polyval(cpf_ply_Z, dkt - cpf0) * 1e-6
    )  # Z component of satellite CPF prediction in megametres

    vX = (
        (np.polyval(cpf_ply_X, dkt2 - cpf0) - np.polyval(cpf_ply_X, dkt1 - cpf0))
        * 1e-6
        * 86400.0
    )  # X velocity component in megametres/day
    vY = (
        (np.polyval(cpf_ply_Y, dkt2 - cpf0) - np.polyval(cpf_ply_Y, dkt1 - cpf0))
        * 1e-6
        * 86400.0
    )  # Y velocity component in megametres/day
    vZ = (
        (np.polyval(cpf_ply_Z, dkt2 - cpf0) - np.polyval(cpf_ply_Z, dkt1 - cpf0))
        * 1e-6
        * 86400.0
    )  # Z velocity component in megametres/day

    ddr = np.sqrt(cX**2 + cY**2 + cZ**2)  # Radial distance to satellite from geo-centre
    dvel = np.sqrt(vX**2 + vY**2 + vZ**2)  # Velocity magnitude
    rv = ddr * dvel

    ql = (cY * vZ - cZ * vY) / rv  # X component of across track from cross products
    qm = (cZ * vX - cX * vZ) / rv  # Y component of across track
    qn = (cX * vY - cY * vX) / rv  # Z component of across track

    dxi = np.array(cX)
    dyi = np.array(cY)
    dzi = np.array(cZ)

    dvx = np.array(vX)
    dvy = np.array(vY)
    dvz = np.array(vZ)

    ddr = np.array(ddr)
    ql = np.array(ql)
    qm = np.array(qm)
    qn = np.array(qn)

    zdum = np.zeros(neps)

    rej2 = 1.0e10  # set initial large rejection level
    rej3 = 1.0e10
    rmsa = 0.0

    itr_fin = False
    # Iteration loop in which the orbit correction parameters are adjusted to give a best fit to the residuals.
    while itr < itrm:
        itr = itr + 1  # Iteration number

        sw = swi
        if itr <= 4:
            sw = 2.0 * swi  # apply loose constrained weighting for early iterations

        ssr = 0.0
        nr = 0
        oldrms = rmsa

        rhs = np.zeros([nu], dtype="longdouble")
        cv = np.zeros([nu], dtype="longdouble")
        rd = np.array(np.zeros([nu, nu], dtype="longdouble"), order="F")

        # apply along-track, across-track and radial corrections to satellite geocentric coordinates. the corrections
        # have been determined from previous iteration, and accumulated values are stored in variables
        # alnc (in days), acrc and radc (in megametres)

        tp = (Depc - dtobc) / 60.0  # Time measured from mid-time of pass
        if Depc[-1] < Depc[0]:
            sel = np.where(Depc > Depc[-1])
            tp[sel] = (Depc[sel] - 86400.0) / 60.0

        # Evaluate constant terms + time rates of change
        # argument minutes of time, measured from mid-time of pass

        al = (
            alnc + alndc * tp + alnddc * tp * tp
        )  # Along track correction from accumulated values, rates and accelerations
        ac = acrc + acrdc * tp + acrddc * tp * tp  # Across track correction
        ra = radc + raddc * tp + radddc * tp * tp  # Radial correction

        # Update XYZ coordinates from the across track, long track and radial corrections
        dx = dxi + dvx * al + ql * ac + (dxi * ra / ddr)
        dy = dyi + dvy * al + qm * ac + (dyi * ra / ddr)
        dz = dzi + dvz * al + qn * ac + (dzi * ra / ddr)

        dxt = (
            dx - STAT_X
        )  # X component difference between satellite and station in megametres
        dyt = dy - STAT_Y  # Y component
        dzt = dz - STAT_Z  # Z component
        dr = np.sqrt(
            dxt * dxt + dyt * dyt + dzt * dzt
        )  # Range from station to satellite
        drc = dr * 2.0  #  2 way range in megametres

        # Calculate the telescope elevation angle for input to the refraction delay model from the satellite altitude
        # relative to geodetic zenith.

        gvs[0] = np.cos(STAT_LATrad) * np.cos(STAT_LONGrad)  # Station X unit vector
        gvs[1] = np.cos(STAT_LATrad) * np.sin(STAT_LONGrad)  # Station Y unit vector
        gvs[2] = np.sin(STAT_LATrad)  # Station Z unit vector
        dstn = np.sqrt(
            gvs[0] * gvs[0] + gvs[1] * gvs[1] + gvs[2] * gvs[2]
        )  # Normalise the unit vectors
        czd = (dxt * gvs[0] + dyt * gvs[1] + dzt * gvs[2]) / (
            dr * dstn
        )  # Zenith height component of SAT->STAT vector / vector range
        altc = np.arcsin(czd) * 360.0 / (2.0 * np.pi)  # inverse sin() to give elevation

        if itr < itrm:
            #  Compute partial derivatives. First, range wrt along-track, across-track and radial errors in the predicted ephemeris.
            drdal = (dvx * dxt + dvy * dyt + dvz * dzt) / dr
            drdac = (ql * dxt + qm * dyt + qn * dzt) / dr
            drdrd = (dx * dxt + dy * dyt + dz * dzt) / (dr * ddr)

        # Time rates of change of these partials are computed by multiplying the above constant quantities by time in
        # minutes from mid-pass time (tp). For accelerations, multiply by tp*tp.
        # These multiplications are carried out below,when the equation of condition vector 'cv' is set up.

        cv = [
            drdal,
            drdac,
            drdrd,
            drdal * tp,
            drdac * tp,
            drdrd * tp,
            drdal * tp * tp,
            drdac * tp * tp,
            drdrd * tp * tp,
            zdum,
            zdum,
            zdum,
            zdum,
        ]

        # Introduce weights. Set SE of a single observation (sw) to 2 cm
        for j in range(9):
            cv[j] = cv[j] / sw

        # Compute refraction delay and apply to predicted 2-way range , using the Marini and Murray model.
        if nmet > 0:
            refr = refmm(
                PRESSURE, TEMP, HUM, altc, Wavel * 1e-3, STAT_LATrad, STAT_HEI_Mm
            )
            delr = refr * 1.0e-6
            drc = drc + delr

        drc = (1e6 * drc / sol) * 1.0e9  # convert computed range to nsecs (2 way)

        tresid = (Drng * 1.0e-3 - drc) / 2.0  # 1 way observational o-c in nsecs
        dresid = (
            tresid * sol * 1e-6 * 1.0e-9
        )  # o-c in Mm for solution and rejection test
        dresid = dresid / sw  # weight residual

        aresid = abs(dresid)
        #  Use data within 3.0*rms for determining rms. Use data within 2.0*rms for solution.

        Ssel = np.where(aresid < rej2)[0]
        rmsb = np.std(dresid[Ssel])
        Rsel = np.where(aresid < rej3)[0]
        rms3 = np.std(dresid[Rsel])
        if itrm - itr < 2:
            Ssel = Rsel

        if itr == 1:
            print(
                "\n\t  #      pts         rms2          rms3          rmsa        TBias      Radial"
            )
            pltresx1 = Depc  # Plot residuals
            pltresy1 = tresid

        rej3 = 3.0 * rms3
        rej2 = 2.0 * rms3

        ssr = np.sum(
            dresid[Ssel] * dresid[Ssel], dtype="longdouble"
        )  # Sum of residual squares
        nr = np.size(Ssel)  # number of residuals

        pltres = tresid * 1.0e3

        # Form normal eqns.
        for j in range(nu):
            rhs[j] = np.sum(cv[j][Ssel] * dresid[Ssel], dtype="longdouble")
            for k in range(nu):
                rd[k, j] = np.sum(cv[j][Ssel] * cv[k][Ssel], dtype="longdouble")

        # Apply A-PRIORI standard errors to unknowns tdot,rdot,tddot,rddot values are stored in data statement.
        if itr < itrm:
            rd[3, 3] = rd[3, 3] + (1.0 / sigt) ** 2
            rd[5, 5] = rd[5, 5] + (1.0 / sigr) ** 2
            rd[6, 6] = rd[6, 6] + (1.0 / sigtt) ** 2
            rd[8, 8] = rd[8, 8] + (1.0 / sigrr) ** 2

            rhs[3] = rhs[3] + (1.0 / sigt) * alnd
            rhs[5] = rhs[5] + (1.0 / sigr) * rdld
            rhs[6] = rhs[6] + (1.0 / sigtt) * alndd
            rhs[8] = rhs[8] + (1.0 / sigrr) * rdldd

            nus = [1, 4, 7]  # Suppress across track unknowns
            rd[nus, :] = 0.0
            rd[:, nus] = 0.0
            rd[nus, nus] = 1.0
            rhs[nus] = 0.0

            nus = [9, 10, 11, 12]  # Suppress pulse-dependent mean values
            rd[nus, :] = 0.0
            rd[:, nus] = 0.0
            rd[nus, nus] = 1.0
            rhs[nus] = 0.0

            # Carry out least squares solution
            ierr, rd = dchols(rd, nu)  #  invert normal matrix
            if ierr != 0:
                print("FAILED to invert normal matrix - quit", ierr)
                sys.exit()

            for i in range(nu):
                rf[i] = 0.0

            # Form solution vector, rf.
            for i in range(nu):
                for j in range(nu):
                    rf[i] = rf[i] + rd[i, j] * rhs[j]

            # Form sum of squares after solution.
            rra = ssr
            for i in range(nu):
                rra = rra - rf[i] * rhs[i]

            if rra < 0.0:
                rra = 0.0

            ins = 3

            rmsa = np.sqrt(rra / nr) * 1.0e6
            seuw = 0.0
            if nr + ins > nu:
                seuw = rra / (1.0 * (nr - nu + ins))

            # Form vector of standard errors, s
            for i in range(nu):
                s[i] = 0.0
                if (rhs[i] != 0.0) & (seuw > 0.0):
                    s[i] = np.sqrt(rd[i, i] * seuw)

            if itr < itrm:
                aln = rf[0]  # along track corrections
                saln = s[0] * 8.64e7

                acr = rf[1]  # across track corrections
                sacr = s[1] * 1.0e6

                rdl = rf[2]  # radial correction
                srdl = s[2] * 1.0e6

                # Get corrections to rates of change of those parameters and their accelerations.

                alnd = rf[3]
                acrd = rf[4]
                rdld = rf[5]
                salnd = s[3] * 8.64e7
                sacrd = s[4] * 1.0e6
                srdld = s[5] * 1.0e6

                alndd = rf[6]
                acrdd = rf[7]
                rdldd = rf[8]
                salndd = s[6] * 8.64e7
                sacrdd = s[7] * 1.0e6
                srdldd = s[8] * 1.0e6

                # Accumulate corrections during iteration

                alnc = alnc + aln
                acrc = acrc + acr
                radc = radc + rdl
                alndc = alndc + alnd
                acrdc = acrdc + acrd
                raddc = raddc + rdld
                alnddc = alnddc + alndd
                acrddc = acrddc + acrdd
                radddc = radddc + rdldd

        print(
            f"\t{itr:3d} {np.size(Ssel):8d}   {1e9 * rmsb * sw:11.3f}   {1e9 * rms3 * sw:11.3f}   {1000.0 * rmsa * sw:11.3f}    {alnc * 8.64e7:9.4f}  {radc * 1.0e6:9.4f}"
        )

        if (abs(oldrms - rmsa) * sw < 0.00001) & (itr >= 10):
            if not itr_fin:
                itrm = itr + 2
                itr_fin = True

    print(
        f"\n\tSatellite orbital time bias (ms)    {alnc * 8.64e7:10.4f} \t{saln:8.4f}"
    )
    print(f"\tSatellite radial error (m)          {radc * 1.0e6:10.4f} \t{srdl:8.4f}")
    print(
        f"\tRate of time bias (ms/minute)       {alndc * 8.64e7:10.4f} \t{salnd:8.4f}"
    )
    print(f"\tRate of radial error (m/minute)     {raddc * 1.0e6:10.4f} \t{srdld:8.4f}")
    print(
        f"\tAcceleration of time bias           {alnddc * 8.64e7:10.4f} \t{salndd:8.4f}"
    )
    print(
        f"\tAcceleration of radial error        {radddc * 1.0e6:10.4f} \t{srdldd:8.4f}"
    )

    if abs(alnc * 8.64e7) > 100.0:
        runWarningList.append(
            "Large Time Bias required " + "{:9.3f}".format(alnc * 8.64e7) + " ms"
        )
        print(
            "\n -- Large Time Bias required " + "{:9.3f}".format(alnc * 8.64e7) + " ms"
        )
    elif abs(alnc * 8.64e7) > 10.0:
        runWarningList.append(
            "Time Bias required " + "{:9.3f}".format(alnc * 8.64e7) + " ms"
        )
        print("\n -- Time Bias required " + "{:9.3f}".format(alnc * 8.64e7) + " ms")

    if abs(radc * 1.0e6) > 100.0:
        runWarningList.append(
            "Large Radial Offset required " + "{:9.3f}".format(radc * 1.0e6) + " m"
        )
        print(
            "\n -- Large Radial Offset required "
            + "{:9.3f}".format(radc * 1.0e6)
            + " m"
        )
    elif abs(radc * 1.0e6) > 10.0:
        print("\n -- Radial Offset required " + "{:9.3f}".format(radc * 1.0e6) + " m")

    presid = tresid * 1e3
    aRMS = np.std(presid)
    aresid = abs(presid)

    print(f"\n\tFlattened range residuals RMS {aRMS:.2f} ps")

    if RRout:
        # write range residuals to a file
        filerr = open("resids.dat", "w")
        for i, ep in enumerate(Depc):
            filerr.write(
                "{:18.12f}".format(ep)
                + " "
                + "{:14.12f}".format(1e-12 * Drng[i])
                + " "
                + "{:18.12f}".format(presid[i] * 1e-3)
                + "\n"
            )
        filerr.close()