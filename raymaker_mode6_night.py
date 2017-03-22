from mpi4py import MPI
import numpy as np
from scipy import interpolate
import os
from partition import partition
import itertools
import random
import time
import datetime as dt
import commands
import subprocess
import shutil
from random import shuffle


from index_helpers import load_TS_params
from index_helpers import load_Dst
from index_helpers import load_Kp
from index_helpers import load_ae
from index_helpers import Kp_at
from index_helpers import Ae_at


import xflib  # Fortran xform-double library (coordinate transforms)
import bisect   # fast searching of sorted lists
from bmodel_dipole import bmodel_dipole
# ------------ Start MPI -------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
host = commands.getoutput("hostname")
if "cluster" in host:
    host_num = int(host[-3:])
elif "nansen" in host:
    host_num = 0


# print "host: %s, host num: %d"%(host, host_num)
nProcs = 1.0*comm.Get_size()


# ------------------Constants --------------------------
R_E = 6371


# -------------- Simulation params ---------------------
t_max = 20     # Maximum duration in seconds

dt0 = 1e-3      # Initial timestep in seconds
dtmax = 5e-2    # Maximum allowable timestep in seconds
root = 2        # Which root of the Appleton-Hartree equation
                # (1 = negative, 2 = positive)
                # (2=whistler in magnetosphere)
fixedstep = 0   # Don't use fixed step sizes, that's a bad idea.
maxerr = 1e-4   # Error bound for adaptive timestepping
maxsteps = 2e5  # Max number of timesteps (abort if reached)
modelnum = 6    # Which model to use (1 = ngo, 2=GCPM, 3=GCPM interp, 4=GCPM rand interp)
use_IGRF = 0    # Magnetic field model (1 for IGRF, 0 for dipole)
use_tsyg = 0    # Use the Tsyganenko magnetic field model corrections
fixed_MLT = 1
MLT = 0

minalt   = (R_E + 800)*1e3 # cutoff threshold in meters

vec_inds = [0, 1, 2, 3, 4]     # Which set of default params to use for the gcpm model
# Kpvec = [0, 2, 4, 6, 8]

dump_model = True
run_rays   = True

ray_out_root ='/shared/users/asousa/WIPP/rays/2d/nightside/mode6/'


# ---------------- Input parameters --------------------
ray_datenum = dt.datetime(2010, 1, 1, 00, 00, 00);

yearday = '%d%03d'%(ray_datenum.year, ray_datenum.timetuple().tm_yday)
milliseconds_day = (ray_datenum.second + ray_datenum.minute*60 + ray_datenum.hour*60*60)*1e3 + ray_datenum.microsecond*1e-3
# Coordinate transformation library
xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/raymaker/libxformd.so')

inp_lats = np.arange(12, 61, 1) #[35] #np.arange(30, 61, 5) #[40, jh41, 42, 43]

# # Get solar and antisolar points:
# sun = xf.gse2sm([-1,0,0], ray_datenum)
# sun_geomag_midnight = xf.sm2rllmag(sun, ray_datenum)
# sun = xf.gse2sm([1,0,0], ray_datenum)
# sun_geomag_noon = xf.sm2rllmag(sun, ray_datenum)
 
# # inp_lons = [sun_geomag_midnight[2]]

# print sun_geomag_noon
# print sun_geomag_midnight
inp_lons = [xf.MLT2lon(ray_datenum, MLT)]


launch_direction = 'up'


launch_alt = (R_E + 1000)*1e3

f1 = 200; f2 = 30000;
num_freqs = 33
flogs = np.linspace(np.log10(f1), np.log10(f2), num_freqs)
freqs = np.round(pow(10, flogs)/10.)*10


# freqs = [200, 500, 1000, 10000, 30000]
# # TESTS FOR SINGLE CASE
# freqs = [18750]
# inp_lats = [33]
# inp_lons = [78]

# Damping parameters:
damp_mode = 1  # 0 for old 2d damping code, 1 for modern code

project_root = '/shared/users/asousa/WIPP/raymaker/'
raytracer_root = '/shared/users/asousa/software/raytracer_DP/'
damping_root = '/shared/users/asousa/software/damping/'
ray_bin_dir    = os.path.join(raytracer_root, 'bin')
# ray_out_dir = os.path.join(project_root, 'rays','dayside','ngo_igrf')


configfile = '/shared/users/asousa/WIPP/raymaker/newray_default.in'


# ------------ Load Kp, Dst, etc at this time -------------

# Mean parameter vals for set Kp:
Kpvec = [0, 2, 4, 6, 8]
Aevec = [1.6, 2.2, 2.7, 2.9, 3.0]
Dstvec= [-3, -15, -38, -96, -215]
Pdynvec=[1.4, 2.3, 3.4, 5.8, 7.7]
ByIMFvec=[-0.1, -0.1, 0.1, 0.5, -0.2]
BzIMFvec=[1.0, 0.6, -0.5, -2.3, -9.2]


# Kp   = Kpvec[vec_ind]
# AE   = Aevec[vec_ind]
# Pdyn = Pdynvec[vec_ind]
# Dst  = Dstvec[vec_ind]
# ByIMF= ByIMFvec[vec_ind]
# BzIMF= BzIMFvec[vec_ind]
# W = np.zeros(6)   # Doesn't matter if we're not using Tsyg


ray_out_dirs = [os.path.join(ray_out_root, 'kp%d'%Kpvec[x]) for x in vec_inds]

# -------- Partition tasks for MPI --------------------
if rank == 0:
    lats, lons, fs, vinds = np.meshgrid(inp_lats, inp_lons, freqs, vec_inds)
    lats = lats.flatten()
    lons = lons.flatten()
    fs   = fs.flatten()
    vinds = vinds.flatten() 

    alts = launch_alt*np.ones_like(lats)
    # alts[fs < 600] += 3000e3

    tasklist = zip(alts, lats, lons, fs, vinds)

    print "%d total tasks"%len(tasklist)
    shuffle(tasklist)
    # tasklist = [(launch_alt, w,x,y) for w,x,y in itertools.product(inp_lats, inp_lons, freqs)]
    chunks = partition(tasklist, nProcs)

else:
    tasklist = None
    chunks = None

tasklist = comm.bcast(tasklist, root=0)
chunks   = comm.bcast(chunks, root=0)

nTasks  = 1.0*len(tasklist)
nSteps = np.ceil(nTasks/nProcs).astype(int)



# print "Subprocess %s on %s thinks tasklist has length %d"%(rank, host, len(tasklist))
# print "Subprocess %s on %s thinks chunks has length %d"%(rank, host, len(chunks))

# ---------- Set up output directory tree -------------
if rank==0:

    for ray_out_dir in ray_out_dirs:
        if (not os.path.exists(ray_out_dir)):
            os.mkdir(ray_out_dir)

        if (not os.path.exists(os.path.join(ray_out_dir, "logs"))):
            os.mkdir(os.path.join(ray_out_dir,"logs"))

        for f in freqs:
            ray_out_subdir = os.path.join(ray_out_dir,'f_%d'%f)
            if (not os.path.exists(ray_out_subdir)):
                os.mkdir(ray_out_subdir)

        for f in freqs:
            for lo in inp_lons:
                ray_out_subdir = os.path.join(ray_out_dir, 'f_%d'%f, 'lon_%d'%lo)
                if (not os.path.exists(ray_out_subdir)):
                    os.mkdir(ray_out_subdir)



# Sync up:
time.sleep(5)
comm.Barrier()

working_path = os.path.join(os.path.expanduser("~"),"rayTmp")



# (if using full GCPM model, you need all the stupid data files in your working directory)
os.chdir(working_path);

# Each subprocess does a subset of frequencies
if run_rays:
    if (rank < len(chunks)):

        print "Subprocess %s on %s: doing %d rays"%(rank, host, len(chunks[rank]))
        for inp in chunks[rank]:
            try:


                vec_ind = inp[4]
                Kp   = Kpvec[vec_ind]
                AE   = Aevec[vec_ind]
                Pdyn = Pdynvec[vec_ind]
                Dst  = Dstvec[vec_ind]
                ByIMF= ByIMFvec[vec_ind]
                BzIMF= BzIMFvec[vec_ind]
                W = np.zeros(6)   # Doesn't matter since we're not using Tsyg

                ray_out_dir = ray_out_dirs[vec_ind]

                # print inp
                ray_out_subdir = os.path.join(ray_out_dir, "f_%d"%inp[3], "lon_%d"%(inp[2]))

                ray_inpfile   = os.path.join(working_path,'ray_inputs_%d_%d_%d.txt'%(inp[3], inp[1], inp[2]))
                ray_outfile   = os.path.join(ray_out_subdir, 'ray_%d_%d_%d.ray'%(inp[3], inp[1], inp[2]))
                ray_tempfile  = os.path.join(working_path,   'ray_%d_%d_%d.ray'%(inp[3], inp[1], inp[2]))
                damp_outfile  = os.path.join(ray_out_subdir,'damp_%d_%d_%d.ray'%(inp[3], inp[1], inp[2]))
                damp_tempfile = os.path.join(working_path,  'damp_%d_%d_%d.ray'%(inp[3], inp[1], inp[2]))

                ray_templog   = os.path.join(working_path,        "ray_%g_%g_%g.log"%( inp[3], inp[1], inp[2])) 
                ray_logfile   = os.path.join(ray_out_dir, "logs", "ray_%g_%g_%g.log"%( inp[3], inp[1], inp[2]))         
                damp_templog  = os.path.join(working_path,        "damp_%g_%g_%g.log"%(inp[3], inp[1], inp[2])) 
                damp_logfile  = os.path.join(ray_out_dir, "logs", "damp_%g_%g_%g.log"%(inp[3], inp[1], inp[2]))


                if not os.path.exists(ray_outfile):
                    print "doing", ray_outfile 

                    # Rotate from geomagnetic to SM cartesian coordinates
                    inp_coords = xf.rllmag2sm(inp[0:3], ray_datenum)
                    # inp_coords = xf.s2c(inp)

                    # Write ray to the input file (used by the raytracer):
                    f = open(ray_inpfile,'w')
                    pos0 = inp_coords
                    if (launch_direction is 'up'):
                        dir0 = pos0/np.linalg.norm(pos0)    # radial outward
                    elif (launch_direction is 'field-aligned'):
                        dir0 = np.zeros(3)                # Field aligned (set in raytracer)

 
                    w0   = inp[3]*2.0*np.pi
                    f.write('%1.15e %1.15e %1.15e %1.15e %1.15e %1.15e %1.15e\n'%(pos0[0], pos0[1], pos0[2], dir0[0], dir0[1], dir0[2], w0))
                    f.close()
                    print "inputs: ", pos0, dir0, w0

                    # --------- Run raytracer --------

                    cmd= '%s/raytracer --outputper=%d --dt0=%g --dtmax=%g'%(ray_bin_dir, 1, dt0, dtmax) + \
                         ' --tmax=%g --root=%d --fixedstep=%d --maxerr=%g'%(t_max, root, fixedstep, maxerr) + \
                         ' --maxsteps=%d --minalt=%d --inputraysfile=%s --outputfile=%s'%( maxsteps, minalt, ray_inpfile, ray_tempfile) + \
                         ' --modelnum=%d --yearday=%s --milliseconds_day=%d'%(modelnum, yearday, milliseconds_day) + \
                         ' --use_tsyganenko=%d --use_igrf=%d --tsyganenko_Pdyn=%g'%(use_tsyg, use_IGRF, Pdyn) + \
                         ' --tsyganenko_Dst=%g --tsyganenko_ByIMF=%g --tsyganenko_BzIMF=%g'%( Dst, ByIMF, BzIMF ) + \
                         ' --tsyganenko_W1=%g --tsyganenko_W2=%g --tsyganenko_W3=%g'%(W[0], W[1], W[2]) + \
                         ' --tsyganenko_W4=%g --tsyganenko_W5=%g --tsyganenko_W6=%g'%(W[3], W[4], W[5]) + \
                         ' --MLT=%g --fixed_MLT=%g'%(xf.lon2MLT(ray_datenum, inp[2]), fixed_MLT)

                    cmd += ' --ngo_configfile=%s'%configfile
                    cmd += ' --kp=%g'%Kp

                    # Run the raytracer
                    print "%s/%d: %s"%(host, rank, cmd)

                    runlog = subprocess.check_output(cmd, shell=True)
                    with open(ray_templog,"w") as file:
                        file.write(runlog)
                        file.close()

                    # ------- Run Damping Code ------------

                    damp_cmd =  '%sbin/damping --inp_file %s --out_file %s '%(damping_root, ray_tempfile, damp_tempfile) + \
                                '--Kp %g --AE %g --mode %d'%(Kp, AE, damp_mode) + \
                                ' --yearday %s --msec %d'%(yearday, milliseconds_day)

                    print "%s/%d: %s"%(host, rank, damp_cmd)

                    damplog = subprocess.check_output(damp_cmd, shell=True)
                    with open(damp_templog,"w") as file:
                        file.write(damplog)
                        file.close()


                    # Move completed files to the output directory
                    if (not os.path.exists(ray_out_subdir)):
                        print "Process %d on %s can't find the path at %s"%(rank, host, ray_out_subdir)
                    else:
                        # shutil.move(ray_tempfile,  ray_outfile)     # ray
                        # shutil.move(damp_tempfile, damp_outfile)    # damp
                        os.system('mv %s %s'%(ray_tempfile,  ray_outfile))
                        os.system('mv %s %s'%(damp_tempfile, damp_outfile))
                        shutil.move(ray_templog,   ray_logfile)     # ray log
                        shutil.move(damp_templog,  damp_logfile)    # damp log

                    os.remove(ray_inpfile)
            except:
                print "Exception at input: ", inp

comm.Barrier()


# Dump plasmasphere model
if (dump_model):

    if rank==0:
        print "Dumping plasmasphere model"
        tasklist = []

        for vind in vec_inds:
            tasklist.append(('XY', vind))
            tasklist.append(('XZ', vind)) 
            tasklist.append(('YZ', vind))
        
        chunks = partition(tasklist, min(nProcs, len(tasklist)))
        print "chunks:"
        print chunks
    else:
        tasklist = None
        chunks = None

    tasklist = comm.bcast(tasklist, root=0)
    chunks   = comm.bcast(chunks, root=0)

    nTasks  = 1.0*len(tasklist)
    nSteps = np.ceil(nTasks/nProcs).astype(int)


    if (rank < len(chunks)):
        for chunk in chunks[rank]:
            plane = chunk[0]
            vec_ind  = chunk[1]


            Kp   = Kpvec[vec_ind]
            AE   = Aevec[vec_ind]
            Pdyn = Pdynvec[vec_ind]
            Dst  = Dstvec[vec_ind]
            ByIMF= ByIMFvec[vec_ind]
            BzIMF= BzIMFvec[vec_ind]
            W = np.zeros(6)   # Doesn't matter since we're not using Tsyg

            ray_out_dir = ray_out_dirs[vec_ind]


            print "process %d doing %s, %d"%(rank, plane, vec_ind)
            
            maxD = 10.0*R_E*1e3
            if plane=='XZ':
                minx = -maxD
                maxx = maxD
                miny = 0
                maxy = 0
                minz = -maxD
                maxz = maxD
                nx = 200
                ny = 1
                nz = 200
            if plane=='XY':
                minx = -maxD
                maxx = maxD
                miny = -maxD
                maxy = maxD
                minz = 0
                maxz = 0
                nx = 200
                ny = 200
                nz = 1
            if plane=='YZ':    
                minx = 0
                maxx = 0
                miny = -maxD
                maxy = maxD
                minz = -maxD
                maxz = maxD
                nx = 1
                ny = 200
                nz = 200


            # model_outfile='model_dump_mode%d_%d_%s.dat'%(modelnum, use_IGRF, plane)
            model_outfile=os.path.join(ray_out_dir, 'model_dump_%s.dat'%(plane))

            cmd = '%s '%os.path.join(ray_bin_dir, 'dumpmodel') +\
                    ' --modelnum=%d --yearday=%s --milliseconds_day=%d '%(modelnum, yearday, milliseconds_day) + \
                    '--minx=%g --maxx=%g '%(minx, maxx) +\
                    '--miny=%g --maxy=%g '%(miny, maxy) +\
                    '--minz=%g --maxz=%g '%(minz, maxz) +\
                    '--nx=%g --ny=%g --nz=%g '%(nx, ny, nz) +\
                    '--filename=%s '%(model_outfile) +\
                    '--use_igrf=%g --use_tsyganenko=%g '%(use_IGRF,0) +\
                    '--tsyganenko_Pdyn=%g '%(Pdyn) +\
                    '--tsyganenko_Dst=%g '%(Dst) +\
                    '--tsyganenko_ByIMF=%g '%(ByIMF) +\
                    '--tsyganenko_BzIMF=%g '%(BzIMF) +\
                    '--tsyganenko_W1=%g '%(W[0]) +\
                    '--tsyganenko_W2=%g '%(W[1]) +\
                    '--tsyganenko_W3=%g '%(W[2]) +\
                    '--tsyganenko_W4=%g '%(W[3]) +\
                    '--tsyganenko_W5=%g '%(W[4]) +\
                    '--tsyganenko_W6=%g '%(W[5]) +\
                    '--gcpm_kp=%g '%(Kp) +\
                    '--ngo_configfile=%s '%configfile

                    # '--ngo_configfile=%s '%(os.path.join(working_path,'newray.in')) +\


            if modelnum==4:
                cmd += '--interp_interpfile=%s '%(interpfile) +\
                    '--scattered_interp_window_scale=%d '%(scattered_interp_window_scale) +\
                    '--scattered_interp_order=%d '%(scattered_interp_order) +\
                    '--scattered_interp_exact=%d '%(scattered_interp_exact) +\
                    '--scattered_interp_local_window_scale=%d '%(scattered_interp_local_window_scale)

            if modelnum==6:
                cmd += ' --kp=%g '%(Kp)
            print cmd

            os.system(cmd)
            # os.system('mv %s %s'%(model_outfile, os.path.join(ray_out_dir, model_outfile)))






comm.Barrier()

if rank==0:
    print "-------- Finished with raytracing ---------"


