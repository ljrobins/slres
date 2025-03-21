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
import time
import os
import subprocess
import sys, getopt
import warnings
import matplotlib

# if (matplotlib.get_backend() != "TkAgg") & (matplotlib.get_backend() != "Qt5Agg"):
#     try:
#         matplotlib.use("TkAgg")
#     except:
#         matplotlib.use("Qt5Agg")

from matplotlib.pyplot import *
import matplotlib.dates as md
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter

import numpy
import scipy
from scipy.stats import skew, kurtosis
from scipy import interpolate
from scipy.optimize import curve_fit
import math
from decimal import *

warnings.simplefilter(action="ignore", category=FutureWarning)


###
# Program Functions
###


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


##
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


##
def mjd2cal(mjd):
    # Return the calendar year, month and day of month from the Modified Julian Day

    mil = 51543.0
    milm = dt.datetime(1999, 12, 31, 0, 0, 0)
    deld = mjd - mil
    ret = milm + dt.timedelta(days=deld)

    return ret.year, ret.month, ret.day


##
def cal2mjd(yr, mm, dd):
    # Return the Modified Julian Day from the calendar year, month and day of month

    mil = 51543.0
    milm = dt.datetime(1999, 12, 31, 0, 0, 0)

    dref = dt.datetime(yr, mm, dd, 0, 0, 0)

    delt = dref - milm
    mjdref = mil + delt.days

    return mjdref


##
def SNXcoords(STAT_id, mjd):
    a = mjd2cal(mjd)
    b = dt.datetime(a[0], a[1], a[2])
    year = b.timetuple().tm_yday
    PVid = open("SLR_PV.snx", "r")
    PVsite = False
    PVxyz = False
    STAT_name = ""
    STAT_X = 0.0
    STAT_Y = 0.0
    STAT_Z = 0.0
    for line in PVid:
        if "-SITE/ID" in line:
            PVsite = False
            if mjd == 0.0:
                break
        if PVsite:
            code = line[1:5]
            if code == STAT_id:
                lon = (
                    float(line[44:47])
                    + float(line[48:50]) / 60.0
                    + float(line[51:55]) / 3600.0
                )
                lat = (
                    float(line[56:59])
                    + float(line[59:62]) / 60.0
                    + float(line[63:67]) / 3600.0
                )
                hei = float(line[68:75])
                STAT_name = line[21:32]
        if "+SITE/ID" in line:
            PVsite = True
            PVid.readline()

        if "-SOLUTION/ESTIMATE" in line:
            PVxyz = False
            break
        if PVxyz:
            code = line[14:18]
            if code == STAT_id:
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
        if "+SOLUTION/ESTIMATE" in line:
            PVxyz = True
            PVid.readline()
    if STAT_name == "":
        print(f"No coordinates in SINEX file for station ID {STAT_id}")
        sys.exit()

    PVid.close()

    if mjd != 0.0:
        a = mjd2cal(mjd)
        tdt = dt.datetime(a[0], a[1], a[2]) + dt.timedelta(seconds=86400.0 * (mjd % 1))

        delt = (tdt - PVdt).total_seconds() / (365.25 * 86400.0)

        STAT_X = STAT_X + delt * VEL_X
        STAT_Y = STAT_Y + delt * VEL_Y
        STAT_Z = STAT_Z + delt * VEL_Z

        STAT_X = STAT_X * 1e-6
        STAT_Y = STAT_Y * 1e-6
        STAT_Z = STAT_Z * 1e-6

    return STAT_name, lat, lon, hei, STAT_X, STAT_Y, STAT_Z


##
def SNXstats(FRdata):
    if not os.path.isfile(FRdata):
        print(f"\t+ File {FRdata} does not exist")
        sys.exit()

    PVid = open("SLR_PV.snx", "r")
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
    if FRdata != "":
        print(f"\t+ Stations in FRD file {FRdata}")
        cmd = f"grep -e h2 -e H2 {FRdata}"
        H2 = os.popen(cmd).read().split("\n")
        for line in H2:
            a = line.split()
            if len(a) > 0:
                if (a[0] == "h2") | (a[0] == "H2"):
                    Fcodes.append(a[2])

        sel = np.where(np.isin(STATcodes, Fcodes))[0]
    else:
        print(f"\t- No FRD file provided. List all stations")

    ni = 0
    STATcodes = np.array(STATcodes)
    STATnames = np.array(STATnames)
    print(f"\n\t  #   Station Code     Station Name       Num Passes")
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
            except:
                if rawi == "q":
                    sys.exit()
                ist = -1

        return STATcodes[sel[ist]]
    else:
        return -1


##
def distPEAK(amp, bins, m1, m3):
    gw = 3.0  # gauss RMS width in ps
    winw = 3  # data window
    x = np.arange(-1.0 * winw, 1.0 * winw + 1.0, 1.0)
    smth = np.exp(-((x / gw) ** 2)) / (gw)  # gauss profile
    smth = smth / np.sum(smth)  # normalise

    profil = np.convolve(amp, smth, mode="same")  # smooth distribution
    pmax = profil.argmax()  # find peak

    ## Fit tangents around the peak of the smoothed profile
    ## to find minumum gradient and adjust peak value
    Prm = 1e6
    Pri = 0
    for si in np.arange(-1 * winw, winw + 1, 1):
        P1 = pmax + si - winw
        P2 = pmax + si + winw + 1
        if (P1 > 0) & (P2 < nbins):
            Px = np.arange(P1, P2, 1)
            Py = profil[P1:P2]
            Pr = np.polyfit(Px, Py, 1)[0]  # find gradient to tangent
            if np.abs(Pr) < Prm:
                Prm = np.abs(Pr)
                Pri = si

    pmax = pmax + Pri

    # figure()
    # f=gcf().number
    # plot(bins,amp,'.-',color="grey")
    # plot(bins,profil,'b.-',label="Smooth")
    # plot(bins[pmax],profil[pmax],'go',label="Peak",ms=6.0)
    # plot(m3,profil[pmax],'ms',label="3*sigma",ms=6.0)
    # plot(m1,profil[pmax],'r*',label="1*sigma",ms=8.0)
    # legend()
    # title(f"Normal Point #{f:02d}")
    # savefig(f"pics/NPhist_{f:02d}.pdf")

    return pmax


##
def sig1PEAK(N):
    if len(N) == 1:
        return N[0]
    rsd = np.std(N)
    rm = np.mean(N)

    rsel = np.where(abs(N - rm) < 3.0 * rsd)[0]
    rsd = np.std(N[rsel])
    m1 = np.mean(N[rsel])

    m0 = m1 + 1.0
    mi = 0
    while (abs(m1 - m0) > 0.001) & (mi < 20):
        mi = mi + 1
        m0 = m1
        rsel = np.where(abs(N - m0) <= 1.0 * rsd)[0]
        m1 = np.mean(N[rsel])

    return m1


###
# Main Program
###

nu = 13  # lsq size
sol = 299792458.0  # Speed of Light m/s
swi = 1.5e-8  # weighting applied to residual fitting
dae = 6.378137000  # PARAMETERS OF SPHEROID
df = 298.2570
CsysID = "--"  # CRD system confirguation ID
ERdata = ""  # Epoch-Range datafile
FRdata = ""  # ILRS full-rate date file in CRD format
CPFin = ""  # ILRS XYZ orbit prediction in CPF format
pstr = ""  # 3-character string to select the prediction provider
METin = ""  # File containing the local meteorological data
CALin = ""  # File containing the system delay values
INmjd = 0.0  # Input the MJD for raw data sets
INcom = 0.0  # Input one-way centre of mass correction in mm.
STAT_id = ""  # File containing the local meteorological data
STAT_name = ""  # Station name
STAT_abv = ""  # Satellite abbreviation
STAT_coords = ""  # Station coordinates string
SATtarget_name = ""  # Satellite name
savepassname = ""  # Save name
frate = 0.0  # Laser fire rate
CPFv1 = False  # Use CPF v1 predictions
SNXref = True  # Flag to acquire station coordinates fron SINEX file
SNXst = False  # Flag to acquire station coordinates fron SINEX file
autoCPF = False  # Flag to attempt to automatically fetch the ILRS CPF file according to the full-rate data filename
METap = False  # Flag to show met data applied
PWidth = 0.0  # Laser pulse width
PWadjust = False  # Adjust Gaussian fit by considering laser pulse width
Wavel = 532.0  # Laser pulse width
plotRES = False  # Flag to plot and save the results, no display
displayRES = False  # Flag to display and save the results
clipGauss = False  # GAUSS
clipsigma = False  # Set residual clipping  wrt to calculated RMS
clipLEHM = False  # Set residual clipping infront and behind the LEHM
clipWIDE = False  # Clip residuals at wider limits for inclusion in full-rate output
peakDIST = False  # Peak determined from smoothed distribution. Default sets Peak as mean from a 1*sigma iteration
NPout = False  # Output normal points to file
NPfull = False  # Output full CRD header and normal points to file
NP50r = False  # Recompute 50 record in CRD output
NP50 = True  # Include a 50 record in CRD output
r50 = ""  # 50 record string
q50 = "0"  # Data quality assessment indicator
RRout = False  # Output range residuals to file
FRout = False  # Output epoch ranges in full-rate CRD
FRin = False  # Input epoch ranges in full-rate CRD
SLVout = False  # Output orbit parameters to file
Unfilter = False  # Flag to include the data in the CRD full-rate data that is flagged as 'noise'
Qpass = False  # Flag to conduct a quick pass on first iteration of fit
ipass = -1  # select pass in FRD file
setupWarningList = []  # setup warnings output in summary
runWarningList = []  # run warnings output in summary
Dchannel = 0  # Detector Channel

NPbin_length = -1.0  # Set length of normal point in seconds
minNPn = -1  # Set the minimum number of range observations to form a Normal point
Xext = ".png"

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
    if opt == "-h":
        print(
            "\n\torbitNP.py - An program to adjust CPF orbits to flatten SLR data and output normal points.\n\n"
            "\tInput an epoch-range data file, either in ILRS FRD format or raw epoch-range data.\n"
            "\The corresponding CPF prediction file is required. This can be automatically fetched from the EDC Data Center.\n"
            "\tFor raw data, include the system delay calibration values and meteorological data.\n\n"
            "\tOptions:\n"
            "\t  -f <FRD file>      \tFull-rate CRD format file\n"
            "\t  -c <CPF file>      \tCPF prediction file\n"
            "\t  -A                 \tAuto find CPF prediction\n"
            "\t  -p <provider>      \tCPF prediction provider (3 character string)\n"
            "\t  -d <datafile>      \tRaw epoch-range data file\n"
            "\t  -m <met file>      \tLocal meteorological datafile\n"
            "\t                     \t     FORMAT: [mjd] [seconds of day] [pressure (mbar)] [temperature (K)] [humidity (%)] e.g.\n "
            "\t                     \t       57395 11638.080 998.32 276.20  87.6\n"
            "\t  -b <cal file>      \tSystem delay calibration to be applied\n"
            "\t                     \t     FORMAT: [mjd] [seconds of day] [two-way system delay in picoseconds] [number of points] [RMS] [skew] [kurtosis] [peak-mean] [survey distance (m)] e.g.\n "
            "\t                     \t       57395 11638.080 99568.32\n"
            "\t                     \t          or\n"
            "\t                     \t       57395 11638.080 99568.32 2365 23.3 0.1 -0.4 1.0 122.977\n"
            "\t  -j <mjd>           \tModified julian day of raw data at start of observations\n"
            "\t  -C <com>           \tOne-way centre of mass correction in mm to improve orbit correction. Not applied to data\n"
            "\t  -s <station code>  \tStation 4-digit code\n"
            "\t  -L                 \tList station codes in supplied FRD file and select. Or all codes in SNX file\n"
            "\t  -I <system ID>     \tSystem Configuration ID for CRD output\n"
            "\t  -t <target name>   \tSatellite target name used to fetch CPF\n"
            '\t  -l <"Lat Lon Alt"> \tString containing station latitude (deg), longitude (deg) and altitude (m)\n'
            "\t  -H <Hz>            \tLaser repetition rate\n"
            "\t  -w <ps>            \tLaser pulse width\n"
            "\t  -W <nm>            \tLaser wavelength. Default 532 nanometres\n"
            "\t  -N <seconds>       \tNormal point length in seconds, Default 30 seconds\n"
            "\t  -M <number>        \tMinimum number of range observations to form a normal point. Default 30\n"
            "\t  -e                 \tInclude unfiltered range measurements\n"
            "\t  -V                 \tSearch for CPF version 1 predictions. Default version 2\n"
            "\t  -q                 \tMake initial solution iterations a quick pass with evenly selected data.\n"
            "\t  -g <factor>        \tApply clipping at factor*gauss sigma from peak of Gauss fit [2.0 - 6.0 permitted]\n"
            "\t  -G <factor>        \tApply clipping at factor*gauss sigma for wider clipping\n"
            "\t  -k <factor>        \tApply factor*sigma clipping to flattened residuals [2.0 - 6.0 permitted]\n"
            "\t  -K <factor>        \tApply factor*sigma clipping for wider clipping\n"
            "\t  -u <lower:upper>   \tApply fixed clipping from the LEHM of the distribution at <lower> and <upper> limits in ps\n"
            "\t  -U <lower:upper>   \tApply fixed clipping from the LEHM for wider clipping at <lower> and <upper> limits in ps\n"
            "\t  -P                 \tDetermine peak from tangent to a smoothed distribution. Default is mean from 1*sigma rejection\n"
            "\t  -o                 \tReturn to pass selection list in FRD file on analysis completion\n"
            "\t  -n                 \tOutput normal points to file normalp.dat in CRD format\n"
            "\t  -F                 \tOutput final epoch and ranges to file fullr.dat in full-rate CRD format\n"
            "\t  -S                 \tRecompute Statistics Record '50' in CRD format\n"
            "\t  -r                 \tOutput one-way O-C range residuals to file resids.dat\n"
            "\t  -v                 \tOutput obit solve parameters to file solvep.out\n"
            "\t  -y                 \tOutput full-rate CRD headings and normal points to fullnormalp.dat\n"
            "\t  -x                 \tPlot final results and save as .png, no display\n"
            "\t  -X                 \tPlot final results, display and save as .png in 'PlotResiduals' folder\n"
            "\t  -z                 \tSpecify plot save filename, without .png extension\n"
            "\t  -Q                 \tOutput PDF file for full quality\n\n"
            "  Examples:\n"
            "\t run orbitNP.py -f lageos1_201606.frd -A -X -s 7237 -N 120 -o -r\n"
            "\t run orbitNP.py -d epochrange.dat -c ajisai_cpf_181010_7831.jax -m met.dat -u -50:300 -U -100:500 -x -s 7840 -N 30 -n -j 58401 -t Ajisai\n\n"
        )
        sys.exit()
    elif opt in ("-f"):  # input file in full-rate CRD format
        FRdata = arg
        FRin = True
    elif opt in ("-c"):  # input satellite prediction file in CPF format
        CPFin = arg
    elif opt in ("-p"):  # input preferred satellite prediction provider
        pstr = arg
    elif opt in ("-d"):  # input epoch-range raw data file
        ERdata = arg
    elif opt in ("-m"):  # input meteorological data
        METin = arg
    elif opt in ("-b"):  # input system calibration data
        CALin = arg
    elif opt in ("-j"):  # input modified julian day for satellite pass
        INmjd = float(arg)
    elif opt in ("-C"):  # input com correction
        INcom = float(arg)
    elif opt in ("-s"):  # input station code
        STAT_id = arg
    elif opt in ("-L"):  # list station codes
        SNXst = True
    elif opt in ("-l"):  # input station coordinates
        STAT_coords = arg
        SNXref = False
    elif opt in ("-I"):  # input system id
        CsysID = arg
    elif opt in ("-t"):  # input satellite target name
        SATtarget_name = arg
    elif opt in ("-H"):  # input laser repetition rate in Hz
        frate = float(arg)
    elif opt in ("-w"):  # input laser pulse width in ps
        PWidth = float(arg)
        PWadjust = True
    elif opt in ("-W"):  # input laser wavelength in ns
        Wavel = float(arg)
    elif opt in ("-N"):  # input normal point length in seconds
        NPbin_length = float(arg)
    elif opt in ("-M"):  # input minimum number of points for a normal point
        minNPn = float(arg)
    elif opt in ("-e"):  # set use of all CRD data points
        Unfilter = True
    elif opt in ("-V"):  # set use CPF v1
        CPFv1 = True
    elif opt in ("-q"):  # set use of 1st quick iteration of orbit fit
        Qpass = True
    elif opt in ("-A"):  # auto fetch CPF files
        autoCPF = True
    elif opt in ("-g"):
        clipGauss = True
        try:
            cfactor = float(arg)
        except:
            print("-g: Failed to read Gauss sigma factor")
            sys.exit()
        if (cfactor < 2.0) | (cfactor > 6.0):
            print(
                "-g: Gauss sigma rejection factor outside of permitted limits [2.0 - 6.0]"
            )
            sys.exit()
    elif opt in ("-G"):  # input N*sigma values for wide clipping
        clipWIDE = True
        try:
            cfactor2 = float(arg)
        except:
            print("-G: Failed to read wide Gauss sigma factor")
            sys.exit()
    elif opt in ("-k"):  # input N*sigma values for clipping
        clipsigma = True
        try:
            cfactor = float(arg)
        except:
            print("-k: Failed to read sigma factor")
            sys.exit()
        if (cfactor < 2.0) | (cfactor > 6.0):
            print("-k: Sigma rejection factor outside of permitted limits [2.0 - 6.0]")
            sys.exit()
    elif opt in ("-K"):  # input N*sigma values for wide clipping
        clipWIDE = True
        try:
            cfactor2 = float(arg)
        except:
            print("-K: Failed to read wide sigma factor")
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
    elif opt in ("-U"):  # input upper and lower values for wide clipping in ps
        clipWIDE = True
        a = arg.split(":")
        if np.size(a) != 2:
            print("-U: Incorrect LEHM lower and upper wide input values")
            sys.exit()
        LEHMlow2 = np.min([float(a[0]), float(a[1])])
        LEHMupp2 = np.max([float(a[0]), float(a[1])])
        if LEHMlow2 > 0.0:
            print("\t ** Clipping not set below the LEHM")
            setupWarningList.append(
                "- Clipping not set below the LEHM, suggest '-U -"
                + str(int(LEHMlow2))
                + ":"
                + str(int(LEHMupp2))
                + "'"
            )
    elif opt in ("-P"):  # set Peak as high point of smotthed distribution
        peakDIST = True
    elif opt in ("-o"):  # set loop through passes in full-rate combined file
        FRDloop = True
    elif opt in ("-n"):  # set normal points output to file
        NPout = True
    elif opt in ("-y"):  # set normal points output to file with CRD headers
        NPfull = True
    elif opt in ("-F"):  # set full-rate output to file
        FRout = True
    elif opt in ("-S"):  # recompute CRD 50 record
        NP50r = True
    elif opt in ("-r"):  # set residuals output to file
        RRout = True
    elif opt in ("-v"):  # set solved orbit parameters output to file
        SLVout = True
    elif opt in ("-x"):  # set plot
        plotRES = True
    elif opt in ("-X"):  # set plot and save
        plotRES = True
        displayRES = True
    elif opt in ("-Q"):  # set plot output file to pdf
        Xext = ".pdf"
    elif opt in ("-z"):  # input plot save filename
        savepassname = arg.split(".")[0]


# Quit if both full-rate file and raw file provided
if (FRin) & (ERdata != ""):
    print("\t- Cannot provide both full-rate and raw laser range data files -quit")
    sys.exit()

# Warning if request for full normal point file outpt with CRD headers but no full-rate file provided
if (NPfull) & (not FRin):
    print("\t- Warning -CRD header normal point file requires full-rate data input")
    setupWarningList.append(
        "- CRD header normal point file requires full-rate data input"
    )
    NPfull = False


if SNXst:
    if STAT_id != "":
        print(f"\t+ Station code already provided as {STAT_id}")
    else:
        print("\t- List SLR Stations")
        STAT_id = SNXstats(FRdata)
        if STAT_id == -1:
            sys.exit()

if not FRin:
    if ERdata == "":
        print("\t- No Epoch-Range datafile provided")
        sys.exit()
    elif ERdata != "":
        print("\t+ Epoch-Range datafile is ", ERdata)
        dataf = ERdata

        if INmjd == 0.0:
            print("\tModified Julian Day not provided")
            sys.exit()
        else:
            print(f"\t+ Modified Julian Day provided as {INmjd}")

        if (autoCPF) & (SATtarget_name == ""):
            print("\t- No Satellite -t name provided, cannot auto-fetch CPF file ")
            sys.exit()

        if METin == "":
            print("\t- No meteorological data provided")
            setupWarningList.append("- No meteorological data provided")
            METap = False
        else:
            print("\t+ Met file is ", METin)

        if CALin == "":
            print("\t- No system delay calibration data provided")
            setupWarningList.append("- No system delay calibration data provided")
        else:
            print("\t+ Cal file is ", CALin)

else:
    print("\t+ FRD file is", FRdata)
    if STAT_id == "":
        print("\t- No Station 4-digit code provided. Use -L for a list of stations")
        sys.exit()

    dataf = FRdata


if STAT_id != "":
    [STAT_name, STAT_LAT, STAT_LONG, STAT_HEI, STAT_X, STAT_Y, STAT_Z] = SNXcoords(
        STAT_id, 0.0
    )
    print(f"\t+ Station is {STAT_id} {STAT_name}")

if (STAT_id == "") & (STAT_coords == ""):
    print("\t- No Station ID or co-ordinates provided")
    sys.exit()

if (SNXref) & (STAT_id == ""):
    print("-s: No Station ID provided")
    sys.exit()
elif (not SNXref) & (STAT_coords == ""):
    print("-l: No Station co-ordinates provided")
    sys.exit()


if CsysID != "--":
    print("\t+ System ID is", CsysID)
else:
    print("\t- System ID not provided")

if SATtarget_name != "":
    print("\t+ SLR Target is", SATtarget_name)


if INcom == 0.0:
    print("\t- Centre of mass offset not provided")
else:
    print(f"\t+ Centre of mass one-way offset provided as {INcom}mm")

if (not autoCPF) & (CPFin == ""):
    print("\t- CPF file not provided")
    sys.exit()

if (autoCPF) & (CPFin != ""):
    print("\t+ CPF file provided as " + CPFin + " not auto fetching CPF")
    autoCPF = False
elif CPFin != "":
    print("\t+ CPF Pred is", CPFin)
elif pstr != "":
    print("\t+ CPF provider selected as", pstr)
if autoCPF:
    print("\t+ Attempt to fetch corresponding CPF from EDC Data Cetre")


if NPbin_length == -1.0:
    print("\t- Normal Point bin length NOT defined, using default 30 seconds")
    setupWarningList.append(
        "- Normal Point bin length NOT defined, using default 30 seconds"
    )
    NPbin_length = 30.0

if minNPn <= 0.0:
    print(
        "\t- Minimum number of observations for a Normal Point NOT defined, using default 30"
    )
    setupWarningList.append(
        "Minimum number of observations for a Normal Point NOT defined, using default 30"
    )
    minNPn = 30

if frate == 0.0:
    print("\t- *** Laser Fire Rate not provided! *** ")
    setupWarningList.append("*** Laser Fire Rate not provided! ***")

if PWidth == 0.0:
    print("\t- *** Laser Pulse Width not provided! *** ")

if PWidth == 0.0:
    print(f"\t+ Laser wavelength is {Wavel} nm")

if Unfilter:
    print("\t+ Include unfiltered range measurements")

if CPFv1:
    print("\t+ Use CPF version 1 predictions")

if Qpass:
    print("\t+ Quick-pass with even data distribution for first orbit sovle iteration")

if (clipLEHM) & (clipsigma):
    print("\t- 3*sigma clipping and LEHM clipping cannot be applied together")
    sys.exit()
elif (clipLEHM) & (clipGauss):
    print("\t- 3* Gausss sigma clipping and LEHM clipping cannot be applied together")
    sys.exit()
elif (clipsigma) & (clipGauss):
    print("\t- 3*Gausss sigma clipping and 3*sigma clipping cannot be applied together")
    sys.exit()
elif clipLEHM:
    print("\t+ Apply clipping from LEHM")
elif clipsigma:
    print("\t+ Apply n*sigma clipping")
elif clipGauss:
    print("\t+ Apply n*Gauss sigma clipping")

if clipWIDE:
    print("\t+ Apply wider clipping for CRD output")
    if clipsigma | clipGauss:
        if cfactor2 < cfactor:
            # Warning if wider clipping levels set less than standard clipping
            print(
                f"\t- Warning - Wide clipping level {cfactor2} should not be set less than standard clipping {cfactor}"
            )
            setupWarningList.append("Wide clipping levels less than standard clipping")

if NPout:
    print("\t+ Output normal points to file normalp.dat in CRD format")

if NPfull:
    print("\t+ Output CRD headers and normal points to file fullnormalp.dat")

if RRout:
    print("\t+ Output range residuals to file resids.dat")

if FRout:
    print("\t+ Output final epoch ranges to file fullr.dat in full-rate CRD format")

if SLVout:
    print("\t+ Output obit solve parameters to file solvep.out")

if displayRES:
    print("\t+ Plot, save and display")
elif plotRES:
    print("\t+ Plot and save, but no display")
if savepassname != "":
    print("\t+ Save plot as ", savepassname + Xext)


if plotRES:
    print("\n -- Plot Results")
    fig = figure(1, figsize=(13, 8))


read1st = True
while loop:  # If a multi-pass full-rate file is used, this loop will allow a quick selection of next pass.
    if FRDloop == False:
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

    if FRout:  ## output full-rate header to normal point file
        filefr = open("fullr.dat", "w")

    if FRin:
        # If the data file provided is a full-rate data file in CRD format read in the data header H4, calibrations,
        # the met data entries and the Epoch-Range data.

        print("\n -- Read FRD file for epochs, ranges and meteorological data... ")
        try:
            fid = open(FRdata)
        except IOError as e:
            print(f"\t Failed to open full-rate data file. {e.strerror} : {FRdata}")
            sys.exit()

        if NPfull:  ## output full-rate header to normal point file
            filefullnp = open("fullnormalp.dat", "w")

        if read1st:  # Read through full-rate data for multiple SLR passes
            h2i = list()
            h2l = list()
            h4l = list()
            c10 = 0
            Fcount = list()
            for l, line in enumerate(fid):
                if "h2" in line or "H2" in line:
                    if c10 > 0:
                        Fcount.append(c10)
                    c10 = 0
                    h2i.append(l)
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
                except:
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
                    if NPfull:
                        filefullnp.write(h1l)
                    if FRout:
                        filefr.write(h1l)

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
                    if CsysID == "--":
                        CsysID = a[3]
                elif (a[0] == "C1") | (a[0] == "c1"):  # read from C1 entry
                    frate = float(a[5])
                elif a[0] == "10":
                    if (Unfilter) | (
                        a[5] != "1"
                    ):  # Take all records or filter out the noise flags
                        ep = np.double(a[1]) / 86400.0
                        if ep1 == -1.0:
                            ep1 = ep
                        if ((ep + 300.0 / 86400.0 < mjd1 - INmjd) | (ep < ep1)) & (
                            mjd_daychange == False
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

                if (NPfull) | (FRout):
                    if (a[0] == "H4") | (a[0] == "h4"):
                        b = line.find(a[1])
                        line = (
                            line[:b] + "1" + line[b + 1 :]
                        )  ## data type is now normal point file
                        if NPfull:
                            filefullnp.write(line)
                        if FRout:
                            filefr.write(line)
                    elif a[0] == "50":
                        if not NP50r:
                            if NPfull:
                                filefullnp.write(line)
                            if FRout:
                                filefr.write(line)
                            r50 = line
                            NP50 = False
                        else:
                            q50 = a[6]  # Data quality assessment indicator
                    elif (a[0] != "30") & (
                        a[0] != "10"
                    ):  # & (a[0] != 'h4') & (a[0] != 'H4'):
                        if NPfull:
                            filefullnp.write(line)
                        if FRout:
                            filefr.write(line)

        fid.close()

    else:
        # If the data file provided is not CRD full-rate data is is read as raw epoch ranges. Additional information is required though
        # data files and the command line

        print("\n -- Read raw epoch-range data file ... ")
        c = mjd2cal(INmjd)
        Hyr = str(c[0])
        Hmon = "{:02d}".format(int(c[1]))
        Hd = "{:02d}".format(int(c[2]))
        Hsat = SATtarget_name
        fid = open(ERdata)
        for line in fid:
            a = line.split()
            ep = np.double(a[0]) / 86400.0
            if ep < ep1:
                INmjd = INmjd + 1.0
                c = mjd2cal(INmjd)
            ep1 = ep
            Dmep.append(INmjd + np.double(a[0]) / 86400.0)
            Depc.append(np.double(a[0]))
            Drng.append(int(float(a[1]) * 1.0e12))
            cep = np.double(a[0])
            datT = dt.datetime(c[0], c[1], c[2]) + dt.timedelta(seconds=cep)
            if cep < Depc[0]:
                datT = datT + dt.timedelta(days=1)

            Ddatet.append(datT)

        fid.close()
        mjd1 = np.min(Dmep)
        mjd2 = np.max(Dmep)

        if METin != "":
            fid = open(METin)
            for line in fid:
                a = line.split()
                Mmep.append(np.double(a[0]) + np.double(a[1]) / 86400.0)
                pressure.append(np.double(a[2]))
                TK.append(np.double(a[3]))
                humid.append(np.double(a[4]))
            fid.close()

        if CALin != "":
            fid = open(CALin)
            for line in fid:
                a = line.split()
                Cmep.append(np.double(a[0]) + np.double(a[1]) / 86400.0)
                Crng.append(np.double(a[2]))
                if len(a) > 3:
                    Cn = float(a[3])
                    Cr = float(a[4])
                    Cs = float(a[5])
                    Ck = float(a[6])
                    Cp = float(a[7])
                    surveyd = float(a[8])
                else:
                    Cn = 0.0
                    Cr = 0.0
                    Cs = 0.0
                    Ck = 0.0
                    Cp = 0.0
                    surveyd = 0.0

                Cnum.append(Cn)
                Crms.append(Cr)
                Cskw.append(Cs)
                Ckurt.append(Ck)
                Cpm.append(Cp)
            fid.close()
            SDapplied = False

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

    if not SNXref:
        a = STAT_coords.split()
        STAT_LAT = float(a[0])  # Station Latitude
        STAT_LONG = float(a[1])  # Station Longitude
        STAT_HEI = float(a[2])  # Station HEIGHT ABOVE SPHEROID IN METRES
    else:
        [STAT_name, STAT_LAT, STAT_LONG, STAT_HEI, STAT_X, STAT_Y, STAT_Z] = SNXcoords(
            STAT_id, mjd1
        )  # Get station coordinates from sinex file

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

    # If flag -A is set, use the FRD filename to automatically retrieve the corresponding CPF prediction file
    if autoCPF:
        crefep = Dmep[0]
        if np.mod(crefep, 1) < 0.005:
            Hd = "{:02d}".format(int(Hd) - 1)
            runWarningList.append(
                "Pass begins soon after midnight, prediction from day earlier used"
            )
            print(
                "\n -- Pass begins soon after midnight, searching for prediction from day earlier"
            )

        print("\n -- Fetching CPF prediction file corresponding to FRD file ... ")
        os.system("rm CPF.list")
        Hsat = Hsat.lower()
        if pstr != "":
            cpf_s = Hsat + "_cpf_" + Hyr[-2:] + Hmon + Hd + "_*." + pstr
        else:
            cpf_s = Hsat + "_cpf_" + Hyr[-2:] + Hmon + Hd + "_*"

        cfol = "CPF/"
        if not os.path.exists(cfol):
            os.makedirs(cfol)

        if CPFv1:
            cVfold = "cpf_predicts"
        else:
            cVfold = "cpf_predicts_v2"
            cmd = f"wget --spider ftp://edc.dgfi.tum.de/pub/slr/{cVfold}/{Hyr}/{Hsat} 2>&1 | grep -q 'File ‘{Hsat}’ exists'"
            if os.system(cmd) != 0:
                cVfold = "cpf_predicts"
                runWarningList.append("No CPF v2 prediction available. Used v1")
                print("\n -- No CPF v2 prediction available. Using CPF v1")

        cmd = f"wget -Nq ftp://edc.dgfi.tum.de/pub/slr/{cVfold}/{Hyr}/{Hsat}/{cpf_s} -P CPF"
        os.system(cmd)
        print(cmd)
        dwl = int(
            subprocess.Popen(
                "ls " + cfol + cpf_s + " | wc -l", shell=True, stdout=subprocess.PIPE
            )
            .communicate()[0]
            .strip()
        )
        cmd = f"ls {cfol}{cpf_s} > CPF.list"
        os.system(cmd)
        cpf_lfi = open("CPF.list")
        for c, cfile in enumerate(cpf_lfi):
            cid = open(cfile.strip())
            for cl in cid:
                if cl[0:2] == "10":
                    a = cl.split()
                    cpf_iep = float(a[2]) + float(a[3]) / 86400.0
                    break

            if crefep - cpf_iep < 0.005:
                dwl = dwl - 1

            cid.close()
        cpf_lfi.close()

        if dwl == 0:
            runWarningList.append(
                "No prediction found for same day from provider '"
                + pstr
                + "'. Alternative used"
            )
            print(
                "\n -- Attempting to find alternative CPF prediction file corresponding to FRD file ..."
            )
            frdt = dt.date(int(Hyr), int(Hmon), int(Hd))
            os.system("rm .listing")
            # cmd="wget --spider --no-remove-listing ftp://edc.dgfi.tum.de/pub/slr/cpf_predicts_v2/"+Hyr+'/'+Hsat+'/'
            cmd = f"wget --spider --no-remove-listing ftp://edc.dgfi.tum.de/pub/slr/{cVfold}/{Hyr}/{Hsat}/"
            os.system(cmd)
            cpf_listing = open(".listing")

            for entry in cpf_listing:
                a = entry.split()
                if a[-1][0] != ".":
                    cpfdate = a[-1].split("_")[-2]
                    cpfdt = dt.date(
                        2000 + int(cpfdate[0:2]), int(cpfdate[2:4]), int(cpfdate[4:6])
                    )
                    if (frdt - cpfdt < dt.timedelta(days=2)) & (frdt >= cpfdt):
                        selpred = a[-1]
                        # cmd = "wget -Nq ftp://edc.dgfi.tum.de/pub/slr/cpf_predicts_v2/"+Hyr+'/'+Hsat+'/'+selpred +" -P CPF"
                        cmd = f"wget -Nq ftp://edc.dgfi.tum.de/pub/slr/{cVfold}/{Hyr}/{Hsat}/{selpred} -P CPF"
                        os.system(cmd)
                        cmd = "ls " + cfol + selpred + ">> CPF.list"
                        os.system(cmd)

            cpf_listing.close()

        numprov = 0
        iprov = 0
        cpf_l = []
        cpf_lfi = open("CPF.list")
        for c, cfile in enumerate(cpf_lfi):
            cid = open(cfile.strip())
            for cl in cid:
                if cl[0:2] == "10":
                    a = cl.split()
                    cpf_iep = float(a[2]) + float(a[3]) / 86400.0
                    break
            cid.close()

            if crefep - cpf_iep > 0.002:
                cpf_l.append(cfile.strip())
                numprov = numprov + 1

        if numprov > 1:
            if pstr != "":
                iprov = numprov - 1
            else:
                print("")
                for c, cfile in enumerate(cpf_l):
                    print("\t", c, "\t", cfile)

                iprov = -1
                print("")
                while (iprov < 0) | (iprov >= numprov):
                    try:
                        rawi = input("  Select prediction provider: ")
                        iprov = int(rawi)
                    except:
                        if rawi == "q":
                            sys.exit()
                        iprov = -1

        if numprov == 0:
            print("No CPF prediction found ")
            sys.exit()

        CPFin = cpf_l[iprov]

    os.system("rm cpf.in")
    # Read CPF Prediction File in the Depc, X, Y, Z lists and produce interpolation functions
    if CPFin == "":
        print(" ERROR: No CPF file")
        sys.exit()
    else:
        with open("cpf.in", "w") as f:
            f.write(CPFin)
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
        except:
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
    rhs = numpy.array(np.zeros(nu), order="F")
    rd = numpy.array(np.zeros([nu, nu]), order="F")
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

    # Calculate the geocentric geocentric co-ordinates of SLR telescope from
    # wrt true equator of date and x-axis same as that of ephemeris
    if not SNXref:
        # compute n+h and (n*b*b/a*a)+h given geodetic co-ordinates and spheroidal parameters a and f.
        db = dae * (1.0 - 1.0 / df)
        dn = dae / (
            np.sqrt(
                np.cos(STAT_LATrad) * np.cos(STAT_LATrad)
                + (db * db * np.sin(STAT_LATrad) * np.sin(STAT_LATrad)) / (dae * dae)
            )
        )  # Earth Radius at Station Lat/LONG
        dnh = dn + STAT_HEI_Mm
        dnab = (dn * db * db) / (dae * dae)
        dnabh = dnab + STAT_HEI_Mm

        STAT_X = dnh * np.cos(STAT_LATrad) * np.cos(STAT_LONGrad)
        STAT_Y = dnh * np.cos(STAT_LATrad) * np.sin(STAT_LONGrad)
        STAT_Z = dnabh * np.sin(STAT_LATrad)

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
        rd = numpy.array(np.zeros([nu, nu], dtype="longdouble"), order="F")

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

        # Apply Centre of Mass Offset in two-way megametres
        if INcom != 0.0:
            drc = drc - 2.0 * INcom * 1.0e-9

        drc = (1e6 * drc / sol) * 1.0e9  # convert computed range to nsecs (2 way)

        tresid = (Drng * 1.0e-3 - drc) / 2.0  # 1 way observational o-c in nsecs
        dresid = (
            tresid * sol * 1e-6 * 1.0e-9
        )  # o-c in Mm for solution and rejection test
        dresid = dresid / sw  # weight residual

        aresid = abs(dresid)
        #  Use data within 3.0*rms for determining rms. Use data within 2.0*rms for solution.

        if (Qpass) & (itr <= 2):
            i = np.arange(np.size(Depc) - 1)
            qd = 0.02 * np.size(i)  # Reduced number of data points evenly to 2%
            if qd < 20:
                qd = 20
            qi = Depc[-1] - Depc[0]
            if qi < 0.0:
                qi = qi + 86400.0

            qg = np.where(Depc[i + 1] - Depc[i] > 2.0)  # Find and remove gaps
            qgs = np.sum(Depc[i[qg] + 1] - Depc[i[qg]])
            qi = qi - qgs

            qi = qi / qd
            qep = (Depc / qi).round() * qi

            Ssel = np.where(qep[i] != qep[i + 1])[
                0
            ]  # Ensure evenly distributed data is selected for orbit

        else:
            Ssel = np.where(aresid < rej2)[0]
            # Ssel=np.where((aresid < rej2) & (altc >30.0))[0]

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

    if SLVout:
        # write solve parameters to file
        fileslv = open("solvep.out", "w")
        fileslv.write(f"{alnc * 8.64e7:15.12f}  {saln:7.3f}\n")
        fileslv.write(f"{radc * 1.0e6:15.12f}  {srdl:7.3f}\n")
        fileslv.write(f"{alndc * 8.64e7:15.12f}  {salnd:7.3f}\n")
        fileslv.write(f"{raddc * 1.0e6:15.12f}  {srdld:7.3f}\n")
        fileslv.write(f"{alnddc * 8.64e7:15.12f}  {salndd:7.3f}\n")
        fileslv.write(f"{radddc * 1.0e6:15.12f}  {srdldd:7.3f}\n")
        fileslv.close()

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

    iRMS = aRMS
    prevRMS = 0.0

    psecbin = 2
    if np.size(presid) < 1500:
        psecbin = 4
    if np.size(presid) < 200:
        psecbin = 8

    pmin = np.min(presid)
    pmax = np.max(presid)
    nbins = int((pmax - pmin) / psecbin)

    i = 0
    i = len(presid)
    while nbins > 10000:
        i = int(i * 0.999)
        pmin = np.min(presid[abs(presid).argsort()[:-i]])
        pmax = np.max(presid[abs(presid).argsort()[:-i]])
        nbins = int((pmax - pmin) / psecbin)
        print(i, nbins, pmax, pmin, psecbin)
    if nbins < 50:
        nbins = 50

    if nbins > 5000:
        print(
            "Large number of histogram bins required.  Plot Processing slow...    "
            + str(nbins)
            + " bins."
        )
        runWarningList.append(
            "Large number of histogram bins required.  Plot Processing slow...    "
            + str(nbins)
            + " bins."
        )

    [hbins, hstep] = np.linspace(pmin, pmax, nbins, retstep=True)

    ## The residauls can now be filtered by clipping at high and low points. This can be carried
    ## out using a Gaussian fit, an iterative mean or from the LEHM point, as selected
    ## on the command line input.

    # Define Gauss model function to fit to distribution data:
    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

    LEHM = 0.0
    PEAK = 0.0

    if clipGauss:
        print("\n -- Clipping using gauss")

        # Construct histogram of residual distribution using pre-defined bins 'hbins'
        histr = np.histogram(presid, hbins)
        amp = histr[0]
        pbins = hbins[1:] - 0.5 * hstep

        # Generate a smoothed distribution profile by averaging neighbouring bins
        winlen = 4  # 4ps smoothing
        smth = np.ones(winlen) / winlen
        hprofil = np.convolve(amp, smth, mode="same")

        maxi = hprofil.argmax()
        amax = hprofil[maxi]
        bmax = hbins[maxi]
        if aRMS > 1000.0:
            aRMS = 1000.0

        # p0 is the initial guess for the Gaussian fitting coefficients (A, mu and sigma)
        p0 = [amax, bmax, aRMS]
        coeffG, var_matrix = curve_fit(gauss, pbins, amp, p0=p0)

        gPEAK = coeffG[1]
        gRMS = coeffG[2]

        l1 = gPEAK - cfactor * gRMS
        l2 = gPEAK + cfactor * gRMS

        if clipWIDE:
            l3 = gPEAK - cfactor2 * gRMS
            l4 = gPEAK + cfactor2 * gRMS
            Osel = np.where((presid > l3) & (presid < l4))[0]
            OA = np.where((presid[Osel] > l1) & (presid[Osel] < l2))[0]
            OB = np.where((presid[Osel] < l1) | (presid[Osel] > l2))[0]
        else:
            Osel = np.where((presid > l1) & (presid < l2))[0]
            OA = np.arange(0, len(Osel), 1)
            OB = OA

    elif clipsigma:
        print("\n -- Clipping at N-sigma")
        iRMS = aRMS
        prevRMS = 0.0
        imean = 0.0
        citr = 0
        print(f"\t    #   Num pts     RMS        Mean")
        while abs(prevRMS - iRMS) > 0.0030:
            citr = citr + 1
            Osel = np.where(abs(presid - imean) < cfactor * iRMS)[0]
            prevRMS = iRMS
            iRMS = np.std(presid[Osel])
            imean = np.mean(presid[Osel])
            print(f"\t {citr:4} {len(Osel):8}   {iRMS:8.3f}   {imean:8.3f}")

        if citr < 5:
            print(
                "Only "
                + str(citr)
                + " iterations in clipping at "
                + str(cfactor)
                + "-sigma"
            )
            runWarningList.append(
                "Only "
                + str(citr)
                + " iterations in clipping at "
                + str(cfactor)
                + "-sigma"
            )

        l1 = imean - cfactor * iRMS
        l2 = imean + cfactor * iRMS

        if clipWIDE:
            l3 = imean - cfactor2 * iRMS
            l4 = imean + cfactor2 * iRMS
            Osel = np.where((presid > l3) & (presid < l4))[0]
            OA = np.where((presid[Osel] > l1) & (presid[Osel] < l2))[0]
            OB = np.where((presid[Osel] < l1) | (presid[Osel] > l2))[0]
        else:
            Osel = np.where((presid > l1) & (presid < l2))[0]
            OA = np.arange(0, len(Osel), 1)
            OB = OA

        PEAK = imean

    elif clipLEHM:
        print("\n -- Clipping at limits from LEHM")

        # Construct histogram of residual distribution using pre-defined bins 'hbins'
        histr = np.histogram(presid, hbins)
        amp = histr[0]
        pbins = hbins[1:] - 0.5 * hstep

        # Generate a smoothed distribution profile by averaging neighbouring bins
        winlen = 6  # 6ps smoothing
        smth = np.ones(winlen) / winlen
        hprofil = np.convolve(amp, smth, mode="same")

        # A histogram fit is performed on the front of the distribution using only
        # the bins in front of the peak as defined by the index 'am'. To set 'am' the
        # 20 highest bins are selected and the RMS of their residual bin values is
        # calculated.  If this RMS is greater than 20ps, the highest indexes are
        # removed until this is reached.  This is done to avoid double peaks in
        # distributions. 'am' is taken as the mean of the remaining indexes.

        am = hprofil.argmax() + 2
        ami = hprofil.argsort()[-20:-1]

        #     print ( '       Peak is', "{:7.5f}".format(np.std(1.0*pbins[ami])), 'ps wide from LEHM to FEHM')
        while (np.size(ami) > 3) & (np.std(1.0 * pbins[ami]) > 5.0):
            #         print ( np.std(1.0*pbins[ami]) ,ami)
            d = ami.argmax()
            ami = np.delete(ami, d)

        am = int(np.mean(1.0 * ami)) + 1

        ar = 0
        af = np.min(np.where(hprofil > 0.25 * np.max(hprofil)))

        ar = am - int(50 / psecbin)
        if PWadjust:
            ar = am - int(3 * PWidth / psecbin)
        if ar > af:
            ar = af
        if ar < 0:
            ar = 0

        if aRMS > 1000.0:
            aRMS = 1000.0
        # p0 is the initial guess for the Gaussian fitting coefficients (A, mu and sigma)
        p0 = [np.max(hprofil), 0.0, aRMS]
        pp = hprofil.argmax()

        # Use Scipy function 'curve_fit' to fit a Gaussian function to the front of the
        # distribution.  If the resulting Gaussian fit peak is not 90%-110% of that of the
        # smoothed profile then adjust 'am' to include additional distribution bins
        # further from the front and repeat fit.

        atmpt = 0
        while atmpt < 12:
            atmpt = atmpt + 1
            try:
                if ar < 0:
                    ar = 0
                coeff, var_matrix = curve_fit(gauss, hbins[ar:am], amp[ar:am], p0=p0)
            except Exception as inst:
                print("Gaussian Fit Function - Except" + str(inst.args))
                coeff = np.zeros(3)

            if ((coeff[0] > 1.05 * hprofil[pp]) | (coeff[0] < 0.95 * hprofil[pp])) & (
                abs(coeff[0] - hprofil[pp]) > 2
            ):  # Check fit height
                am = am + 2
                ar = ar - 1
                print(
                    "\t- Front Gaussian Fit - Too Tall/Short - Attempt #" + str(atmpt),
                    coeff,
                )

            elif (
                (coeff[2] < 0.0) | (coeff[1] < pmin) | (coeff[1] > pmax)
            ):  # Check fit coefficients
                am = am + 1
                ar = ar + 1
                print("\t- Front Gaussian Fit - Error - Attempt #" + str(atmpt))

            elif (PWadjust) & (
                coeff[2] > 2.5 * PWidth
            ):  # Use laser pulse width to assess front Gauss fit
                am = am - 1
                ar = ar - 1
                print("\t- Front Gaussian Fit - Too Wide - Attempt #" + str(atmpt))

            else:
                atmpt = 100

            if ar < 0:
                ar = 0

        if (coeff[0] > 1.05 * hprofil[pp]) | (coeff[0] < 0.95 * hprofil[pp]):
            runWarningList.append("Front Gaussian Fit - Too Tall/Short")

        if (PWadjust) & (coeff[2] > 2.5 * PWidth):
            print("- Front Gaussian Fit RMS above limit for 10ps laser pulse")
            runWarningList.append(
                "Front Gaussian Fit RMS above limit for 10ps laser pulse"
            )

        if coeff[0] == 0.0:
            print("- Front Gaussian Fit Fail. " + str(neps) + " Observations")
            runWarningList.append(
                "Front Gaussian Fit Fail. " + str(neps) + " Observations"
            )
            l1 = LEHMlow
            l2 = LEHMupp
            Osel = np.where(aresid)[0]
            OA = np.arange(0, len(Osel), 1)
            OB = OA
        else:
            # Determine the Gaussian fit peak
            PEAK = coeff[1]
            PEAKi = abs(hbins - PEAK).argmin()
            PEAKm = coeff[0]

            # Determine the Gaussian fit leading edge half maximum
            gauss_hist_fit = gauss(hbins, *coeff)
            l = abs(gauss_hist_fit[0:PEAKi] - 0.5 * PEAKm).argsort()[0:6]
            IntpG = interpolate.interp1d(
                gauss_hist_fit[l],
                hbins[l] + 0.5 * hstep,
                kind="linear",
                fill_value="extrapolate",
            )
            LEHM = float(IntpG(0.5 * PEAKm))

            # Determine the histogram profile falling edge half maximum
            l = abs(hprofil[am:-1] - 0.5 * PEAKm).argsort()[0:6] + am
            IntpG = interpolate.interp1d(
                hprofil[l], pbins[l], kind="linear", fill_value="extrapolate"
            )
            FEHM = float(IntpG(0.5 * PEAKm))

            # Set residual clipping points and select data
            l1 = LEHM + LEHMlow
            l2 = LEHM + LEHMupp
            if clipWIDE:
                l3 = LEHM + LEHMlow2
                l4 = LEHM + LEHMupp2
                Osel = np.where((presid > l3) & (presid < l4))[0]
                OA = np.where((presid[Osel] > l1) & (presid[Osel] < l2))[0]
                OB = np.where((presid[Osel] < l1) | (presid[Osel] > l2))[0]
            else:
                Osel = np.where((presid > l1) & (presid < l2))[0]
                OA = np.arange(0, len(Osel), 1)
                OB = OA

    else:
        Osel = np.where(aresid)[0]
        OA = np.arange(0, len(Osel), 1)
        OB = OA

    OUTresid = presid[Osel]  # Select the data within the clipping
    OUTrng = Drng[Osel]
    OUTep = Depc[Osel]
    OUTdt = Ddatet[Osel]
    OUTmjd = Dmep[Osel]

    NORMresid = OUTresid[OA]  # Select the data to form the normal points
    NORMrng = OUTrng[OA]
    NORMep = OUTep[OA]
    NORMdt = OUTdt[OA]
    NORMmjd = OUTmjd[OA]

    PLUSresid = OUTresid[OB]  # Select the data NOT used for normal points
    PLUSdt = OUTdt[OB]

    FILTERflag = np.zeros(np.size(OUTresid))  # CRD filter flag
    FILTERflag[OA] = 2
    FILTERflag[OB] = 1

    tRMS = np.std(NORMresid)

    print(f"\n\tClipped range residuals RMS {tRMS:.2f} ps")

    fbin = NORMep[0] - np.mod(NORMep[0], NPbin_length)
    fdtbin = NORMdt[0] - dt.timedelta(seconds=np.mod(NORMep[0], NPbin_length))
    lbin = NORMep[-1]
    if lbin > fbin:
        NPbins = np.arange(fbin, lbin, NPbin_length)
    else:
        b1 = np.arange(fbin, 86400.0, NPbin_length)
        b2 = np.arange(0.0, lbin, NPbin_length)
        NPbins = np.concatenate((b1, b2))

    NPdtbins = np.array(
        [fdtbin + dt.timedelta(seconds=i * NPbin_length) for i in range(len(NPbins))]
    )

    if NP50:
        # Compute the CRD 50 statistics record
        print(f"\n\t+ Compute CRD 50 Statitics record")
        Fstd = 2.0 * np.std(NORMresid)
        Fskw = skew(NORMresid, axis=0, bias=True)
        Fkrt = kurtosis(NORMresid, axis=0, fisher=True, bias=True)
        av = np.mean(NORMresid)
        m3 = np.mean(NORMresid)
        m1 = sig1PEAK(NORMresid)
        PEAK = m1
        if peakDIST:
            histr = np.histogram(NORMresid, hbins)
            bs = 0.5 * (hbins[1] - hbins[0])
            pi = distPEAK(histr[0], hbins[:-1] + bs, m1, m3)
            PEAK = hbins[pi] + bs
            # print(b,m1,hbins[pi]+bs,m3)

        P50 = PEAK
        r50 = f"50 {CsysID} {Fstd:.1f} {Fskw:.3f} {Fkrt:.3f} {2.0 * (PEAK - av):.1f} {q50}\n"
        if NPfull:
            filefullnp.write(r50)

    if FRout:
        # Output the epochs and ranges and the calibration adn meteorological data in CRD format
        frd_event = 2  #  Epoch event
        frd_detect = Dchannel  #  Detector channel
        frd_stop = 0  #  Stop number
        frd_recva = 0  #  Receive amplitude
        frd_transa = 0  #  Transmit amplitude
        if not FRin:
            for i, ep in enumerate(Mmep):
                sec = (ep % 1) * 86400.0
                filefr.write(f"20 {sec:.3f} {pressure[i]} {TK[i]} {humid[i]} 0\n")
            for i, ep in enumerate(Cmep):
                sec = (ep % 1) * 86400.0
                filefr.write(
                    f"40 {sec:.12f} 0 {CsysID} {Cnum[i]} {Cnum[i]} {surveyd} {Crng[i]} 0.0 {Crms[i]} {Cskw[i]} {Ckurt[i]} {Cpm[i]} 2 0 0 0.0\n"
                )
        if NP50:
            filefr.write(r50)
        for i, ep in enumerate(OUTep):
            filefr.write(
                f"10 {ep:.12f} {1e-12 * OUTrng[i]:.12f} {CsysID}  {frd_event} {FILTERflag[i]:1.0f} {frd_detect} {frd_stop} {frd_recva} {frd_transa}\n"
            )

        filefr.write("H8\nH9\n")
        filefr.close()

    Nav = list()
    Nstd = list()
    Nskw = list()
    Nkrt = list()
    Npk = list()
    Nep = list()
    Nmjd = list()
    Ndatet = list()
    NRng = list()
    Nnpts = list()
    Ndur = list()

    print("\n -- Form Normal Points")
    # calculate normal points from residuals
    for b in NPbins:
        sel = np.where((NORMep >= b) & (NORMep < b + NPbin_length))[0]
        if np.size(sel) >= minNPn:
            av = np.mean(NORMresid[sel])
            Nav.append(av)
            Nstd.append(np.std(NORMresid[sel]))
            Nskw.append(skew(NORMresid[sel], axis=0, bias=True))
            Nkrt.append(kurtosis(NORMresid[sel], axis=0, fisher=True, bias=True))

            m3 = np.mean(NORMresid[sel])
            m1 = sig1PEAK(NORMresid[sel])
            PEAK = m1
            if peakDIST:
                histr = np.histogram(NORMresid[sel], hbins)
                bs = 0.5 * (hbins[1] - hbins[0])
                pi = distPEAK(histr[0], hbins[:-1] + bs, m1, m3)
                PEAK = hbins[pi] + bs
                print(b, m1, hbins[pi] + bs, m3)

            Npk.append(PEAK)
            # print (f"PEAK {PEAK} {av}")

            Nc = abs(
                NORMep[sel] - np.mean(NORMep[sel])
            ).argmin()  # find closest NORMep epoch to NP mean epoch
            Nep.append(NORMep[sel[Nc]])
            Nmjd.append(NORMmjd[sel[Nc]])
            Ndatet.append(NORMdt[sel[Nc]])
            # create NP range values from mean epoch range and NP average residual
            NRng.append(
                NORMrng[sel[Nc]] * 1e-12 + 2.0 * (av - NORMresid[sel[Nc]]) * 1e-12
            )  # form NP range
            Nnpts.append(np.size(sel))
            maxNep = np.max(NORMep[sel])
            minNep = np.min(NORMep[sel])
            if minNep > maxNep:
                maxNep = maxNep + 86400.0
            if maxNep - minNep > 0:  # 04/22/20 rlr
                Ndur.append(maxNep - minNep)
            else:
                # print ("\n maxNep, minNep=", maxNep, minNep)
                Ndur.append(NPbin_length)  # 1 point npt => very low rate
            if np.size(sel) > 1:
                Nf = np.polyfit(NORMep[sel], NORMresid[sel], 1)
                if (abs(Nf[0]) > 20.0 / NPbin_length) & (np.size(sel) > 200):
                    print(
                        f"\t- Slope {Nf[0]:.4f} ps/s for normal point at epoch {NORMep[sel[Nc]]:.1f}"
                    )

        elif np.size(sel) > 0:
            print(f"\t- Normal point at {b} rejected - too few points {np.size(sel)}")

    Nav = np.array(Nav)
    Nstd = np.array(Nstd)
    Nep = np.array(Nep)
    NRng = np.array(NRng)
    Nnpts = np.array(Nnpts)
    Ndur = np.array(Ndur)
    nN = np.size(Nep)

    # use laser fire rate to estimate return rate for NP. Instead of normal point
    # length, use the interval between the latest and earliest epochs in the normal point bin.
    if frate != 0.0:
        rR = 100.0 * Nnpts / (Ndur * frate)
        # rR=100.0*Nnpts/(NPbin_length*frate)
    else:
        rR = np.zeros(np.size(Nnpts))

    if nN == 0:
        print("\n ** No Normal Points to output")
        setupWarningList.append("- No Normal Points were formed from the range data")
    elif NPout | NPfull:
        # print ( normal point results and write to file)
        print("\n -- Write Normal Points")

        if NPout:
            filenp = open("normalp.dat", "w")
            c = mjd2cal(np.floor(Nmjd[0]))
            h = int(Nep[0] / 3600.0)

        for i, epo in enumerate(Nep):
            print(
                f"\t11 {epo:.12f} {NRng[i]:.12f} {CsysID} 2 {NPbin_length:.1f} {Nnpts[i]} {2.0 * Nstd[i]:.1f} {Nskw[i]:.3f} {Nkrt[i]:.3f} {2.0 * (Npk[i] - Nav[i]):.1f} {rR[i]:.1f} {Dchannel} 0.0"
            )
            if NPout:
                filenp.write(
                    f"11 {epo:.12f} {NRng[i]:.12f} {CsysID} 2 {NPbin_length:.1f} {Nnpts[i]} {2.0 * Nstd[i]:.1f} {Nskw[i]:.3f} {Nkrt[i]:.3f} {2.0 * (Npk[i] - Nav[i]):.1f} {rR[i]:.1f} {Dchannel} 0.0\n"
                )

            if NPfull:
                filefullnp.write(
                    f"11 {epo:.12f} {NRng[i]:.12f} {CsysID} 2 {NPbin_length:.1f} {Nnpts[i]} {2.0 * Nstd[i]:.1f} {Nskw[i]:.3f} {Nkrt[i]:.3f} {2.0 * (Npk[i] - Nav[i]):.1f} {rR[i]:.1f} {Dchannel} 0.0\n"
                )

        if NPout:
            filenp.close()
        if NPfull:
            filefullnp.write("H8\nH9\n")
            filefullnp.close()

    # Plot the orbit ranges, range residuals from the orbit and residuals
    # from the adjusted orbit with the distribution histogram
    if plotRES:
        print("\n -- Plot Results")
        if fignum_exists(1):
            fig.clf()
            fig.set_size_inches(13, 8)
        else:
            fig = figure(1, figsize=(13, 8))

        figtext(
            0.05,
            0.97,
            "Satellite Laser Range data from: " + dataf,
            fontsize=13,
            color="#113311",
        )
        figtext(
            0.05,
            0.942,
            "Station: "
            + STAT_abv
            + " "
            + STAT_id
            + "                              Satellite: "
            + SATtarget_name.capitalize(),
            fontsize=12,
            color="#227722",
        )
        figtext(
            0.975,
            0.925,
            "CPF: " + CPFin.split("/")[-1],
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
        )
        if METap == False:
            figtext(
                0.975,
                0.965,
                "No Range Correction Applied From Local Meteorological Data",
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
                color="#bb1111",
            )
        t = figtext(
            0.965,
            0.62,
            "Solve RMS: " + "{:6.2f}".format(aRMS) + "ps",
            fontsize=10,
            horizontalalignment="right",
            color="#227722",
        )
        t.set_bbox(dict(facecolor="#ffffff", alpha=0.65, edgecolor="#ffffff"))
        t = figtext(
            0.965,
            0.5951,
            "Final RMS: " + "{:6.2f}".format(tRMS) + "ps",
            fontsize=10,
            horizontalalignment="right",
            color="#227722",
        )
        t.set_bbox(dict(facecolor="#ffffff", alpha=0.65, edgecolor="#ffffff"))
        t = figtext(
            0.082,
            0.62,
            "Time Bias:   " + "{:7.2f}".format(alnc * 8.64e7) + "ms",
            fontsize=10,
            horizontalalignment="left",
            color="#227722",
        )
        t.set_bbox(dict(facecolor="#ffffff", alpha=0.65, edgecolor="#ffffff"))
        t = figtext(
            0.082,
            0.5951,
            "Radial Offset:  " + "{:7.2f}".format(radc * 1e6) + "m",
            fontsize=10,
            horizontalalignment="left",
            color="#227722",
        )
        t.set_bbox(dict(facecolor="#ffffff", alpha=0.65, edgecolor="#ffffff"))

        if clipLEHM | clipGauss | clipsigma:
            t = figtext(
                0.765,
                0.09,
                f"Clipping:  {l1:.1f}ps and {l2:.1f}ps",
                fontsize=8,
                horizontalalignment="left",
                color="#558855",
            )
            t.set_bbox(dict(facecolor="#ffffff", alpha=0.65, edgecolor="#ffffff"))

        # set up grid for output plots
        gs1 = gridspec.GridSpec(1, 5)
        gs1.update(left=0.075, right=0.975, top=0.92, bottom=0.705, wspace=0.8)
        ax1 = fig.add_subplot(gs1[0, 0:2])
        ax2 = fig.add_subplot(gs1[0, 2:5])
        gs2 = gridspec.GridSpec(1, 4)
        gs2.update(left=0.075, right=0.975, top=0.65, bottom=0.075, wspace=0.05)
        ax3 = fig.add_subplot(gs2[0, 0:3])
        ax4 = fig.add_subplot(gs2[0, 3], sharey=ax3)
        ax1.tick_params(axis="both", which="major", labelsize=8)
        ax2.tick_params(axis="both", which="major", labelsize=8)
        ax3.tick_params(
            "x",
            length=4,
            width=0.5,
            bottom=True,
            top=True,
            direction="in",
            which="minor",
            color=[0.65, 0.65, 0.65],
            labelsize=7,
        )
        ax3.tick_params(
            "x",
            length=12,
            width=1.5,
            bottom=True,
            top=True,
            direction="in",
            which="major",
            color=[0.5, 0.5, 0.5],
            labelsize=8,
        )
        ax3.tick_params(axis="y", which="major", labelsize=9)
        ax4.tick_params(
            axis="y",
            which="major",
            length=4,
            left=True,
            right=True,
            direction="out",
            labelsize=0,
            labelcolor="#ffffff",
        )
        ax4.yaxis.set_visible(True)
        ax4.xaxis.set_visible(False)

        # set time axis limits
        xmin = np.min(Ddatet)
        xmax = np.max(Ddatet)

        # set residaul plot y-axis limits
        ymin = np.min(OUTresid) - 1.0 * aRMS
        ymax = np.max(OUTresid) + 1.0 * aRMS
        if ymax - ymin < 100:
            ymax = 50.0
            ymin = -50.0
            ax3.set_yticks(np.arange(ymin, ymax + 1, 10))

        # plot 2-way ranges vs time
        ax1.ticklabel_format(useOffset=False)
        ax1.plot(
            Ddatet, Drng * 1.0e-12, ".", ms=3.0, mew=0.0, color="#333333", alpha=0.7
        )
        ax1.tick_params(axis="y", labelcolor="#111111", labelsize=8)
        ax1.set_ylabel("Range (s)", fontsize=8, color="#111111")
        ax1.set_xlabel("Epoch", fontsize=8)

        ax1B = ax1.twinx()
        ax1B.plot(Ddatet, altc, ".", ms=3.0, mew=0.0, color="#119966", alpha=0.7)
        ax1B.tick_params(axis="y", labelcolor="#119966", labelsize=8)
        ax1B.set_yticks(np.arange(0, 90 + 1, 10))

        ax1B.set_ylabel("Elevation (°)", fontsize=8, color="#119966")

        ax1B.set_xlim(xmin, xmax)
        ax1B.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        if np.size(ax1B.get_xticks()) > 10:
            ax1B.tick_params(axis="x", which="major", labelsize=7)
            ax1B.tick_params(axis="x", which="major", labelrotation=15)

        # plot residuals from CPF orbit
        ax2.ticklabel_format(useOffset=False)

        ax2.plot(Ddatet, pltresy1, ".", ms=1.0, color="#119966", alpha=0.5)
        ax2.set_xlim(xmin, xmax)
        if np.size(ax2.get_xticks()) > 12:
            ax2.tick_params(axis="x", which="major", labelsize=7)
            ax2.tick_params(axis="x", which="major", labelrotation=15)
        ax2.set_ylabel("CPF Residual (ns)", fontsize=8)
        ax2.set_xlabel("Epoch", fontsize=8)

        # plot final range residuals against time
        minor_ticks = np.array(
            [
                fdtbin + dt.timedelta(seconds=i * 0.1 * NPbin_length)
                for i in range(10 * len(NPbins))
            ]
        )
        if len(minor_ticks) > 200:
            minor_ticks = np.array(
                [
                    fdtbin + dt.timedelta(seconds=i * 0.25 * NPbin_length)
                    for i in range(4 * len(NPbins))
                ]
            )
        ax3.ticklabel_format(useOffset=False)
        ax3.set_xticks(minor_ticks, minor=True)
        ax3.set_xlim(xmin, xmax)

        # control x axis labels at NP start values and limit the number plotted
        if len(NPbins) < 40:
            ax3.set_xticks(NPdtbins)
        else:
            sel = np.arange(0, len(NPdtbins), 2)
            ax3.set_xticks(NPdtbins[sel])

        ntik = len(ax3.get_xticklabels())
        if ntik > 25:
            for l, label in enumerate(ax3.get_xticklabels()):
                if np.mod(l, int(ntik / 25.0)) != 0:
                    label.set_visible(False)
                else:
                    label.set_rotation(25)

        elif ntik > 15:
            # ax3.set_xticks(rotation=25)
            ax3.tick_params(axis="x", which="major", labelrotation=25)

        if NPbin_length > 60.0:
            ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
            ax3.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        else:
            ax2.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
            ax3.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))

        dtdel = OUTdt[1:] - OUTdt[:-1]
        seld = np.where(dtdel < dt.timedelta(seconds=1))
        avdd = 1e6
        if np.size(seld) > 0:
            avdd = np.mean(dtdel[seld]).microseconds

        # plot range residuals and normal point results on chart
        mark = 4.0
        if len(OUTdt) < 2000:
            mark = 6.0
        if (len(OUTdt) > 10000) & (avdd < 100000):
            mark = 3.0
        if len(OUTdt) > 100000:
            mark = 2.0
        if len(OUTdt) > 200000:
            mark = 1.0
        ax3.plot(NORMdt, NORMresid, ".", ms=mark, mew=0.0, color="#000000", alpha=0.6)
        if clipWIDE:
            ax3.plot(
                PLUSdt, PLUSresid, ".", ms=mark, mew=0.0, color="#999999", alpha=0.7
            )
        ax3.set_ylim(ymin, ymax)
        ax3.set_ylabel("Fit Residual (ps)", fontsize=9)
        D = Ddatet[0]
        ax3.set_xlabel(
            "Epoch on date " + "{0}/{1}/{2}".format(D.day, D.month, D.year), fontsize=9
        )

        if NPout | NPfull:
            ax3.errorbar(
                Ndatet,
                Nav,
                Nstd,
                fmt=".",
                ms=0.0,
                ecolor="#11cb11",
                elinewidth=1.7,
                capsize=5,
                capthick=3,
                zorder=3,
            )
            ax3.plot(
                Ndatet,
                Nav,
                "s",
                ms=6.0,
                mew=2.0,
                mec="#11cb11",
                color="#b9ffa9",
                alpha=0.95,
                zorder=3,
            )

        # plot horizontal histogram of range residuals
        if clipWIDE:
            histr = ax4.hist(
                OUTresid,
                hbins,
                orientation="horizontal",
                edgecolor="#779977",
                color="#dddddd",
                lw=0.6,
                alpha=0.5,
            )

        histr = ax4.hist(
            NORMresid,
            hbins,
            orientation="horizontal",
            edgecolor="#003300",
            color="#119966",
            lw=0.4,
        )

        if NP50:
            Pxi = abs(histr[1] - P50).argmin()
            Px = np.mean([histr[0][Pxi - 5 : Pxi + 5]])
            ax4.plot(
                Px, P50, ">", mec="#af309f", mew=1.5, color="#ffffff", ms=6.0, alpha=0.6
            )

        if clipGauss:
            # Plot Gauss fit with coefficients
            hmax = np.max(histr[0])
            gauss_hist_fit = gauss(hbins, *coeffG)
            ax4.plot(gauss_hist_fit, hbins, "-", lw=1.5, color="#55ffff")
            ax4.plot(coeffG[0], coeffG[1], "s", color="#55ffff", ms=6.0)

        if clipLEHM:
            # plot smoothed distribution profile. plot the fitted Gaussian profile. plot peak value. plot LEHM and FEHM
            ax4.plot(hprofil, pbins, "k-", lw=1.1)
            if coeff[0] != 0.0:
                ax4.plot(
                    gauss_hist_fit, hbins + 0.5 * hstep, "-", lw=1.5, color="#ff5533"
                )
                ax4.plot(coeff[0], coeff[1], "s", color="#ff5533", ms=6.0)
                ax4.plot(0.5 * PEAKm, LEHM, "s", color="#f0f000", ms=6.0)
                ax4.plot(0.5 * PEAKm, FEHM, "s", color="#ffffd0", ms=6.0)

        if clipLEHM | clipGauss | clipsigma:
            # plot limit lines for user defined data clipping
            hmax = np.max(histr[0])
            ax4.plot([0.0, hmax], [l1, l1], "-", color="#555555", lw=1.0)
            ax4.plot([0.0, hmax], [l2, l2], "-", color="#555555", lw=1.0)

        # save image to file. Use ipass index if multi pass FR data file used
        pfol = f"PlotResiduals/{Hyr}/"
        if not os.path.exists(pfol):
            os.makedirs(pfol)
        if savepassname != "":
            if ipass == -1:
                pltsav = pfol + savepassname + Xext
            else:
                pltsav = pfol + savepassname + "_" + str(ipass) + Xext
        elif FRin:
            if ipass == -1:
                tstr = FRdata.split("/")[-1][:-4]
                if tstr.find(STAT_id) != 0:
                    pltsav = pfol + STAT_id + "_" + FRdata.split("/")[-1][:-4] + Xext
                else:  # Handle case in which we are looking at individual passes
                    pltsav = pfol + FRdata.split("/")[-1][:-4] + Xext
            else:
                pltsav = (
                    pfol
                    + STAT_id
                    + "_"
                    + FRdata.split("/")[-1][:-4]
                    + "_"
                    + str(ipass)
                    + Xext
                )
        else:
            pltsav = (
                pfol
                + STAT_id
                + "_"
                + SATtarget_name
                + "_"
                + Ddatet[0].strftime("%Y%m%d_%H_00")
                + Xext
            )

        fig.set_size_inches(13, 8)
        print(" -- Save plot ", pltsav)
        savefig(pltsav, dpi=200)

    if NPout:
        print(" -- Save normal points to normalp.dat")
    if NPfull:
        print(" -- Save full CRD normal points to fullnormalp.dat")
    if FRout:
        print(" -- Save full-rate records in CRD to fullr.dat")
    if RRout:
        print(" -- Save residuals to resids.dat")

    if displayRES:
        get_current_fig_manager().set_window_title(pltsav.split("/")[-1][:-4])
        get_current_fig_manager().window.geometry("1300x830")
        show(block=False)

    # print ( warnings summary list to screen    )
    print("\n -- Summary: ")
    wList = np.concatenate((setupWarningList, runWarningList))
    if len(wList) == 0:
        print("\t Completed OK")
    else:
        for line in wList:
            print("\tWarning: ", line)
        if loop:
            rawi = input(
                "\n ** Warnings in process, hit Enter to continue\n                            (q to quit)\n  "
            )
            if rawi == "q":
                sys.exit()
    print("\n ")
