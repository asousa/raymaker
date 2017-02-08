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
t_max = 15      # Maximum duration in seconds

dt0 = 1e-3      # Initial timestep in seconds
dtmax = 1e-1    # Maximum allowable timestep in seconds
root = 2        # Which root of the Appleton-Hartree equation
                # (1 = negative, 2 = positive)
                # (2=whistler in magnetosphere)
fixedstep = 0   # Don't use fixed step sizes, that's a bad idea.
maxerr = 5e-3   # Error bound for adaptive timestepping
maxsteps = 1e5  # Max number of timesteps (abort if reached)
modelnum = 1    # Which model to use (1 = ngo, 2=GCPM, 3=GCPM interp, 4=GCPM rand interp)
use_IGRF = 1    # Magnetic field model (1 for IGRF, 0 for dipole)
use_tsyg = 0    # Use the Tsyganenko magnetic field model corrections

minalt   = (R_E + 500)*1e3 # cutoff threshold in meters

vec_ind = 0     # Which set of default params to use for the gcpm model

dump_model = True
run_rays   = True

ray_out_dir = '/shared/users/asousa/WIPP/rays/ngo_igrf'

# ---------------- Input parameters --------------------
ray_datenum = dt.datetime(2010, 1, 1, 00, 00, 00);

yearday = '%d%03d'%(ray_datenum.year, ray_datenum.timetuple().tm_yday)
milliseconds_day = (ray_datenum.second + ray_datenum.minute*60 + ray_datenum.hour*60*60)*1e3 + ray_datenum.microsecond*1e-3
# Coordinate transformation library
xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/raymaker/libxformd.so')

inp_lats = np.arange(12, 61, 2) #[35] #np.arange(30, 61, 5) #[40, jh41, 42, 43]
# inp_lats = [10,12,14]
# Get solar and antisolar points:
sun = xf.gse2sm([-1,0,0], ray_datenum)
sun_geomag_midnight = np.round(xf.sm2rllmag(sun, ray_datenum))
sun = xf.gse2sm([1,0,0], ray_datenum)
sun_geomag_noon = np.round(xf.sm2rllmag(sun, ray_datenum))


# Nightside
inp_lons_night = np.arange(sun_geomag_midnight[2] - 20, sun_geomag_midnight[2] + 20, 2)
inp_lons_day   = np.arange(sun_geomag_noon[2] - 20,     sun_geomag_noon[2] + 20,     2)

inp_lons = np.hstack([inp_lons_night, inp_lons_day])



launch_alt = (R_E + 1000)*1e3

f1 = 200; f2 = 30000;
num_freqs = 33
flogs = np.linspace(np.log10(f1), np.log10(f2), num_freqs)
freqs = np.round(pow(10, flogs)/10.)*10


# # TESTS FOR SINGLE CASE
# freqs = [18750]
# inp_lats = [33]
# inp_lons = [78]

# Damping parameters:
damp_mode = 1  # 0 for old 2d damping code, 1 for modern code

project_root = '/shared/users/asousa/WIPP/raymaker/'
raytracer_root = '/shared/users/asousa/software/raytracer_v1.17/'
damping_root = '/shared/users/asousa/software/damping/'
ray_bin_dir    = os.path.join(raytracer_root, 'bin')
# ray_out_dir = os.path.join(project_root, 'rays','dayside','ngo_igrf')

# GCPM grid to use (plasmasphere model)
if modelnum==1:
    configfile = os.path.join(project_root,'ngo_infile.in')
# interpfile = os.path.join(project_root,'raytracer_runscripts','gcpm_models','gcpm_kp40_20010101_0000_MLD01.txt')
if modelnum==3:
    interpfile = os.path.join(project_root,'gcpm_models','demo_models','gcpm_kp4_2001001_L10_80x80x80_noderiv.txt')
if modelnum==4:
    interpfile = os.path.join(project_root, 'gcpm_models','demo_models','gcpm_kp4_2001001_L10_random_5000_20000_0_200000_600000.txt')
    scattered_interp_window_scale = 1.5
    scattered_interp_order = 2
    scattered_interp_exact = 0
    scattered_interp_local_window_scale = 5


# ------------ Load Kp, Dst, etc at this time -------------

# Mean parameter vals for set Kp:
Kpvec = [0, 2, 4, 6, 8]
Aevec = [1.6, 2.2, 2.7, 2.9, 3.0]
Dstvec= [-3, -15, -38, -96, -215]
Pdynvec=[1.4, 2.3, 3.4, 5.8, 7.7]
ByIMFvec=[-0.1, -0.1, 0.1, 0.5 -0.2]
BzIMFvec=[1.0, 0.6, -0.5, -2.3, -9.2]


Kp   = Kpvec[vec_ind]
AE   = Aevec[vec_ind]
Pdyn = Pdynvec[vec_ind]
Dst  = Dstvec[vec_ind]
ByIMF= ByIMFvec[vec_ind]
BzIMF= BzIMFvec[vec_ind]
W = np.zeros(6)   # Doesn't matter if we're not using Tsyg



# -------- Partition tasks for MPI --------------------
if rank == 0:
    lats, lons, fs = np.meshgrid(inp_lats, inp_lons, freqs)
    lats = lats.flatten()
    lons = lons.flatten()
    fs   = fs.flatten()

    alts = launch_alt*np.ones_like(lats)
    # alts[fs < 600] += 3000e3

    tasklist = zip(alts, lats, lons, fs)

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
    if (not os.path.exists(ray_out_dir)):
        os.mkdir(ray_out_dir)

    if (not os.path.exists(os.path.join(ray_out_dir, "logs"))):
        os.mkdir(os.path.join(ray_out_dir,"logs"))

    # clean any existing logs
    # os.system("rm %s/*.log"%(os.path.join(ray_out_dir,"logs")))

    for f in freqs:
        ray_out_subdir = os.path.join(ray_out_dir,'f_%d'%f)
        if (not os.path.exists(ray_out_subdir)):
            os.mkdir(ray_out_subdir)

    for f in freqs:
        for lo in inp_lons:
            ray_out_subdir = os.path.join(ray_out_dir, 'f_%d'%f, 'lon_%d'%lo)
            if (not os.path.exists(ray_out_subdir)):
                os.mkdir(ray_out_subdir)

            # clean any existing files
            # os.system("rm %s/*"%ray_out_subdir)








if rank == 0:
    print "Dst: ", Dst
    print "Kp:  ", Kp
    print "Ae:  ", AE


# Sync up:
time.sleep(5)
comm.Barrier()

working_path = os.path.join(os.path.expanduser("~"),"rayTmp")


# working_path = "/tmp";

# (if using full GCPM model, you need all the stupid data files in your working directory)
os.chdir(working_path);

# Each subprocess does a subset of frequencies
if run_rays:
    if (rank < len(chunks)):
        # working_path = os.path.join(os.path.expanduser("~"),"rayTmp_%d"%(rank))
        print "Subprocess %s on %s: doing %d rays"%(rank, host, len(chunks[rank]))
        for inp in chunks[rank]:
            try:
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
                    # # print "Cleaning previous runs..."
                    # if os.path.exists(ray_inpfile):
                    #     os.remove(ray_inpfile)
                    # if os.path.exists(ray_outfile):
                    #     os.remove(ray_outfile)

                    # Rotate from geomagnetic to SM cartesian coordinates
                    inp_coords = xf.rllmag2sm(inp, ray_datenum)
                    # inp_coords = xf.s2c(inp)

                    # Write ray to the input file (used by the raytracer):
                    f = open(ray_inpfile,'w')
                    pos0 = inp_coords
                    # dir0 = pos0/np.linalg.norm(pos0)    # radial outward

                    Bo   = bmodel_dipole(inp)
                    Bo[0]= -1.*np.abs(Bo[0])  # Southern hemi points down... 
                    Bo_t = xf.transform_data_sph2car(inp[1], inp[2], Bo)
                    Bo_c = xf.mag2sm(Bo_t, ray_datenum)
                    dir0 = Bo_c/(-1.0*np.linalg.norm(Bo_c))  # Parallel to B (dipole model)

                    # print pos0/np.linalg.norm(pos0)
                    # print dir0
                    # dir0 = -1.0*Bo_c/np.linalg.norm(Bo_c)
                    # print dir0

                    # dir0 = [0, 0, 1] # [pos0[0], pos0[1], 0]/(np.sqrt(pos0[0]**2 + pos0[1]**2))
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
                         ' --tsyganenko_W4=%g --tsyganenko_W5=%g --tsyganenko_W6=%g'%(W[3], W[4], W[5])

                    # Append model-specific parameters to the command line
                    if  modelnum == 1:
                        cmd += ' --ngo_configfile=%s'%configfile
                    elif modelnum == 2:
                        cmd += ' --gcpm_kp=%g'%Kp
                    elif modelnum == 3: 
                        cmd += ' --interp_interpfile=%s'%interpfile
                    elif modelnum == 4:
                        cmd += ' --interp_interpfile=%s'%interpfile
                        cmd += ' --scattered_interp_window_scale=%g'%scattered_interp_window_scale
                        cmd += ' --scattered_interp_order=%d'%scattered_interp_order
                        cmd += ' --scattered_interp_exact=%d'%scattered_interp_exact
                        cmd += ' --scattered_interp_local_window_scale=%g'%scattered_interp_local_window_scale


                    # Run the raytracer

                    print "%s/%d: %s"%(host, rank, cmd)



                    # use this syntax to write logfiles as we go
                    # file = open(ray_templog, "w+")
                    # subprocess.call(cmd, shell=True, stdout=file)
                    # file.close()


                    runlog = subprocess.check_output(cmd, shell=True)
                    with open(ray_templog,"w") as file:
                        file.write(runlog)
                        file.close()

                    # ------- Run Damping Code ------------

                    damp_cmd =  '%sbin/damping --inp_file %s --out_file %s '%(damping_root, ray_tempfile, damp_tempfile) + \
                                '--Kp %g --AE %g --mode %d'%(Kp, AE, damp_mode) + \
                                ' --yearday %s --msec %d'%(yearday, milliseconds_day)

                    print "%s/%d: %s"%(host, rank, damp_cmd)

                    # os.system(damp_cmd)
                    # file = open(damp_templog, "w+")
                    # subprocess.call(damp_cmd, shell=True, stdout=file)
                    # file.close()

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
        tasklist = ['XZ', 'XY', 'YZ']
        chunks = partition(tasklist, min(nProcs, len(tasklist)))
    else:
        tasklist = None
        chunks = None

    tasklist = comm.bcast(tasklist, root=0)
    chunks   = comm.bcast(chunks, root=0)

    nTasks  = 1.0*len(tasklist)
    nSteps = np.ceil(nTasks/nProcs).astype(int)


    if (rank < len(chunks)):
        plane = chunks[rank][0]
        print "process %d doing %s"%(rank, plane)
        
        maxD = 5.0*R_E*1e3
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
        model_outfile='model_dump_%s.dat'%(plane)

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

        print cmd

        os.system(cmd)
        os.system('mv %s %s'%(model_outfile, os.path.join(ray_out_dir, model_outfile)))






comm.Barrier()

if rank==0:
    print "-------- Finished with raytracing ---------"


