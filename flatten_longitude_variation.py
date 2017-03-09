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


base_dir = '/shared/users/asousa/WIPP/rays/2d/nightside/'

dirs = ['gcpm_kp4']

for d in dirs:
    qsub_cmd = 'qsub -v BASE=%s,INP=%s /shared/users/asousa/WIPP/raymaker/flat_job.pbs'%(base_dir, d)
    print qsub_cmd
    os.system(qsub_cmd)


# Copy damping files:

# for subdir in dirs:
#     ray_dir = os.path.join(base_dir, subdir)
#     d = os.listdir(ray_dir)
#     freqs = sorted([int(f[2:]) for f in d if f.startswith('f_')])
#     d = os.listdir(os.path.join(ray_dir, 'f_%d'%freqs[0]))
#     lons = sorted([int(f[4:]) for f in d if f.startswith('lon_')])
#     d = os.listdir(os.path.join(ray_dir, 'f_%d'%freqs[0], 'lon_%d'%lons[0]))
#     lats = sorted([int(s.split('_')[2]) for s in d if s.startswith('ray_')])

#     for f in freqs:
#         for lon in lons:
#             for lat in lats:
#                 sourcefile = os.path.join(ray_dir, 'f_%d'%f, 'lon_%d'%lon, 'damp_%d_%d_%d.ray'%(f, lat, lon))
#                 destfile   = os.path.join(base_dir + subdir + '_flat', 'f_%d'%f, 'lon_%d'%lon, 'damp_%d_%d_%d.ray'%(f, lat, lon))
#                 print sourcefile
#                 # print destfile
#                 shutil.copy(sourcefile, destfile)