#! env python

# This script imports data for the STELLA FastWaves stencil (saved using
# SerialBox format) and converts it into a NPZ compressed archive.
# Input data and reference output data are expected to come from two different
# datasets.
# The SerialBox serializer software suite is required.
#
# Original data for testing the FastWaves GridTools4Py stencil against a
# STELLA-coded stencil were taken from
# /scratch/daint/jenkins/data/double/oldFW/
import sys
import numpy as np
from serialization import serializer, savepoint


# Parse command line arguments
args = sys.argv[1:]
if len(args) != 5 or args[0] in ["-h","--help"]:
    print("convert_STELLA_FastWaves_data.py expects 5 arguments:")
    print("    1 - directory of SerialBox database for INPUT data")
    print("    2 - name of .json organizer file for INPUT data")
    print("    3 - directory of SerialBox database for REFERENCE RESULTS data")
    print("    4 - name of .json organizer file for REFERENCE RESULTS data")
    print("    5 - name of output archive")
    sys.exit()

input_dir = args[0]
input_json_filename = args[1]
results_dir = args[2]
results_json_filename = args[3]
outfile = args[4]

# Import savepoints from STELLA data using SerialBox
ser = serializer(input_dir, input_json_filename, 'r')
a = [sp for sp in ser.savepoints if 'FastWavesRK' in sp.name]
sp_const = ser.savepoints[0]
spin = a[6]
spout = a[7]

# 3D stencil inputs
u_pos          = ser.load_field('u', spin).squeeze()
v_pos          = ser.load_field('v', spin).squeeze()
utens_stage    = ser.load_field('suten', spin).squeeze()
vtens_stage    = ser.load_field('svten', spin).squeeze()
ppuv           = ser.load_field('zpi', spin).squeeze()
rho            = ser.load_field('rho', spin).squeeze()
wgtfac         = ser.load_field('wgtfac', spin).squeeze()

# Reset domain
domain = u_pos.shape

# Single plane stencil inputs
cwp            = np.zeros(domain, dtype=np.float64)
xdzdx          = np.zeros(domain, dtype=np.float64)
xdzdy          = np.zeros(domain, dtype=np.float64)
xlhsx          = np.zeros(domain, dtype=np.float64)
xlhsy          = np.zeros(domain, dtype=np.float64)
wbbctens_stage = np.zeros((domain[0],domain[1],domain[2]+1),
                          dtype=np.float64)

cwp[:,:,-1]            = ser.load_field('cwp', spin).squeeze()
xdzdx[:,:,-1]          = ser.load_field('xdzdx', spin).squeeze()
xdzdy[:,:,-1]          = ser.load_field('xdzdy', spin).squeeze()
xlhsx[:,:,-1]          = ser.load_field('xlhsx', spin).squeeze()
xlhsy[:,:,-1]          = ser.load_field('xlhsy', spin).squeeze()
wbbctens_stage[:,:,-1] = ser.load_field('wbbctens_stage', spin).squeeze()

# Constant field inputs
rho0 = ser.load_field('rho0', sp_const).squeeze()
p0   = ser.load_field('p0',   sp_const).squeeze()
hhl  = ser.load_field('hhl',  sp_const).squeeze()
acrlat0  = ser.load_field('acrlat',  sp_const).squeeze()[:,0]

# Load STELLA reference results using a second serializer object
serstella = serializer(results_dir, results_json_filename, 'r')
sp_stella = serstella.savepoints[0]
stella_u = serstella.load_field('u', sp_stella).squeeze()
stella_v = serstella.load_field('v', sp_stella).squeeze()

np.savez_compressed(outfile,
         u_pos=u_pos,
         v_pos=v_pos,
         utens_stage=utens_stage,
         vtens_stage=vtens_stage,
         ppuv=ppuv,
         rho=rho,
         wgtfac=wgtfac,
         cwp=cwp,
         xdzdx=xdzdx,
         xdzdy=xdzdy,
         xlhsx=xlhsx,
         xlhsy=xlhsy,
         wbbctens_stage=wbbctens_stage,
         rho0=rho0,
         p0=p0,
         hhl=hhl,
         acrlat0=acrlat0,
         ref_u=stella_u,
         ref_v=stella_v
        )
