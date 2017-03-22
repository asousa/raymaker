import numpy as np

import os

import itertools
import random
import time

import commands

import shutil
from random import shuffle
import xflib
import datetime as dt

ray_datenum = dt.datetime(2010, 1, 1, 00, 00, 00);

# Coordinate transformation library
xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/3dWIPP/python/libxformd.so')


REMOVE_SMALL_FILES = False

inp_lats = np.arange(12, 61, 1) #[35] #np.arange(30, 61, 5) #[40, jh41, 42, 43]
# inp_lats = [10,12,14]
# Get solar and antisolar points:
sun = xf.gse2sm([-1,0,0], ray_datenum)
sun_geomag_midnight = (xf.sm2rllmag(sun, ray_datenum))
sun = xf.gse2sm([1,0,0], ray_datenum)
sun_geomag_noon = (xf.sm2rllmag(sun, ray_datenum))


# Nightside
inp_lons_night = np.arange(sun_geomag_midnight[2] - 20, sun_geomag_midnight[2] + 20, 2)
inp_lons_day   = np.arange(sun_geomag_noon[2] - 20,     sun_geomag_noon[2] + 20,     2)

# inp_lons = np.hstack([inp_lons_night, inp_lons_day])
# inp_lons = [sun_geomag_midnight[2]]
inp_lons = [256]

f1 = 200; f2 = 30000;
num_freqs = 33
flogs = np.linspace(np.log10(f1), np.log10(f2), num_freqs)
freqs = np.round(pow(10, flogs)/10.)*10

print "total files to look for: ", len(inp_lons)*len(inp_lats)*len(freqs)

project_root = '/shared/users/asousa/WIPP/raymaker/'
raytracer_root = '/shared/users/asousa/software/raytracer_v1.17/'
damping_root = '/shared/users/asousa/software/damping/'
ray_bin_dir    = os.path.join(raytracer_root, 'bin')
ray_out_dir = '/shared/users/asousa/WIPP/rays/2d/dayside/gcpm_kp4_flat'

missing_rays = 0
missing_damp = 0
for freq in freqs:
    for lat in inp_lats:
        for lon in inp_lons:
            ray_out_subdir = os.path.join(ray_out_dir, "f_%d"%freq, "lon_%d"%(lon))
            ray_outfile   = os.path.join(ray_out_subdir, 'ray_%d_%g_%g.ray'%(freq, lat, lon))
            damp_outfile  = os.path.join(ray_out_subdir,'damp_%d_%g_%g.ray'%(freq, lat, lon))



            if not (os.path.exists(ray_outfile)):
                print "missing rayfile at %d, %d, %d"%(freq, lat, lon)
                missing_rays += 1
            else:
                raysize = os.stat(ray_outfile).st_size
                if raysize < 8e3:
                    print "file:", ray_outfile, "size: ", raysize
                    if REMOVE_SMALL_FILES:
                        os.system('rm %s'%ray_outfile)
                        os.system('rm %s'%damp_outfile)
            if not (os.path.exists(damp_outfile)):
                print "missing dampfile at %d, %d, %d"%(freq, lat, lon)
                missing_damp += 1

print "Missing %d rays"%missing_rays
print "Missing %d damp"%missing_damp
