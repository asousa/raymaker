import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
import pickle

from scipy import interpolate
import matplotlib.pyplot as plt
import os
import itertools
import random
import os
import time
import datetime as dt

from spacepy import coordinates as coord
from spacepy.time import Ticktock

from raytracer_utils import readdump, read_rayfile, read_rayfiles
from mpl_toolkits.mplot3d import Axes3D

from raytracer_utils import readdump
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --------------- Latex Plot Beautification --------------------------
fig_width_pt = 650.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = 12  # width in inches
fig_height = 4      # height in inches
fig_size =  [fig_width+1,fig_height+1]
params = {'backend': 'ps',
          'axes.labelsize': 14,
          'font.size': 14,
          'legend.fontsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
# --------------- Latex Plot Beautification --------------------------
import xflib  # Fortran xform-double library (coordinate transforms)

# Coordinate transformation library
xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/3dWIPP/python/libxformd.so')




H_IONO=1000e3

# 2d topdown plots
# ----- Plot rays (3d version) -----

# Load rayfile:
ray_roots = ['/shared/users/asousa/WIPP/rays/2d/nightside/gcpm_kp2_flat/']
             # '/shared/users/asousa/WIPP/rays/2d/nightside/gcpm_kp4_flat/',
             # '/shared/users/asousa/WIPP/rays/2d/nightside/gcpm_kp6_flat/',
             # '/shared/users/asousa/WIPP/rays/2d/nightside/gcpm_kp8_flat/']
# ray_root = '/shared/users/asousa/WIPP/rays/2d/nightside/gcpm_kp0_flat/'
clims = [-2, 5]
psize = 10


for ray_root in ray_roots:
  print "doing ", ray_root
  fig_dir = os.path.join(ray_root, 'figures')

  if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

  d = os.listdir(ray_root)
  freqs = [int(f[2:]) for f in d if f.startswith('f_')]

  for rayF in freqs:

    print rayF, "Hz"
    lat_min = 0
    lat_max = 90
    lon_min = 0
    lon_max = 360

    # figtitle = 'mode1_igrf_1k.png'

    rf = read_rayfiles(ray_root,rayF, lat_min, lat_max, lon_min, lon_max)
    plasma_model_dump = os.path.join(ray_root, 'model_dump_XY.dat')
    d_xy = readdump(plasma_model_dump)
    plasma_model_dump = os.path.join(ray_root, 'model_dump_XZ.dat')
    d_xz = readdump(plasma_model_dump)
    plasma_model_dump = os.path.join(ray_root, 'model_dump_YZ.dat')
    d_yz = readdump(plasma_model_dump)

    Ne_xy = d_xy['Ns'][0,:,:,:].squeeze().T*1e-6
    Ne_xy[np.isnan(Ne_xy)] = 0
    Ne_xz = d_xz['Ns'][0,:,:,:].squeeze().T*1e-6
    Ne_xz[np.isnan(Ne_xz)] = 0
    Ne_yz = d_yz['Ns'][0,:,:,:].squeeze().T*1e-6
    Ne_yz[np.isnan(Ne_yz)] = 0

    px = np.linspace(-10, 10, 200)
    py = np.linspace(-10, 10, 200)


    flashtime = dt.datetime(2001, 1, 1, 0, 0, 0)
    R_E = 6371e3 # Radius of earth in meters
    D2R = (np.pi/180.0)

    # Convert to geographic coordinates for plotting:
    rays = []
    for r in rf:
        tmp_coords = coord.Coords(zip(r['pos'].x, r['pos'].y, r['pos'].z),'SM','car',units=['m','m','m'])
        tvec_datetime = [flashtime + dt.timedelta(seconds=s) for s in r['time']]
        tmp_coords.ticks = Ticktock(tvec_datetime) # add ticks
    #     tmp_coords = tmp_coords.convert('MAG','car')
        tmp_coords.sim_time = r['time']

        rays.append(tmp_coords)

    #     derp = tmp_coords[0]
    #     derp = derp.convert('MAG','sph')
    #     print derp
        

    # -------- 2D Plots -------------------
    fig, ax = plt.subplots(1,3)

    # ax[0].set_aspect("equal")
    # ax[1].set_aspect("equal")
    # ax[2].set_aspect("equal")

    # print np.shape(Ne_xy)
    # print np.shape(Ne_xz)
    # print np.shape(Ne_yz)

    # Plot background plasma (equatorial slice)
    p0 = ax[0].pcolormesh(px,py,np.log10(Ne_xy))
    p0.set_clim(clims)
    p1 = ax[1].pcolormesh(px,py,np.log10(Ne_xz))
    p1.set_clim(clims)
    p2 = ax[2].pcolormesh(px,py,np.log10(Ne_yz))
    p2.set_clim(clims)



    # divider = make_axes_locatable(ax[2])
    # cax = divider.append_axes("right",size="4%",pad=0.15)
    # cb = plt.colorbar(p2, cax=cax)
    # cb.set_label('Electron Density (#/cm$^3$)')
    # cticks = np.arange(clims[0],clims[1] + 1)
    # cb.set_ticks(cticks)
    # cticklabels = ['$10^{%d}$'%k for k in cticks]
    # cb.set_ticklabels(cticklabels)




    # Plot the earth
    for i in [0, 1, 2]:
        earth = plt.Circle((0,0),1,color='0.5',alpha=1, zorder=100)
        iono  = plt.Circle((0,0),(R_E + H_IONO)/R_E, color='w',alpha=0.8, zorder=99)
        ax[i].add_patch(earth)   
        ax[i].add_patch(iono)
        
    # Plot rays:
    for r in rays:
        ax[0].plot(r.x/R_E, r.y/R_E, linewidth=3)
        if r.y[0] < 0:
            ax[1].plot(r.x/R_E, r.z/R_E, linewidth=3, zorder=101)
        else:
            ax[1].plot(r.x/R_E, r.z/R_E, linewidth=3, zorder=10)
        if r.x[0] > 0:
            ax[2].plot(r.y/R_E, r.z/R_E, linewidth=3, zorder=101)
        else:
            ax[2].plot(r.y/R_E, r.z/R_E, linewidth=3, zorder=10)

    # Get direction to sun (GSE system - x axis points to sun)
    x_in = [1, 0, 0]
    sun = xf.gse2sm(x_in, flashtime)
    # sun = xf.sm2mag(x_in, flashtime)

    ax[0].plot([0, sun[0]], [0, sun[1]],'w', linewidth=2, zorder=101)
    ax[1].plot([0, sun[0]], [0, sun[2]],'w', linewidth=2, zorder=101)
    ax[2].plot([0, sun[1]], [0, sun[2]],'w', linewidth=2, zorder=101)

            


    ax[0].set_title('XY')
    ax[1].set_title('XZ')
    ax[2].set_title('YZ')
    ax[1].set_yticks([])
    ax[2].set_yticks([])

    ax[1].set_xlabel('L (R$_E$)')
    ax[0].set_ylabel('L (R$_E$)')

    ax[0].set_xlim([-psize, psize])
    ax[0].set_ylim([-psize, psize])
    ax[1].set_xlim([-psize, 0])
    ax[1].set_ylim([-psize/2, psize/2])
    ax[2].set_xlim([-psize, psize])
    ax[2].set_ylim([-psize, psize])

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')

    fig.tight_layout()

    fig.subplots_adjust(right=0.84)
    cax = fig.add_axes([0.85,0.19, 0.01, 0.629])
    cb = plt.colorbar(p2, cax=cax)
    cb.set_label('Electron Density (#/cm$^3$)')
    cticks = np.arange(clims[0],clims[1] + 1)
    cb.set_ticks(cticks)
    cticklabels = ['$10^{%d}$'%k for k in cticks]
    cb.set_ticklabels(cticklabels)

    fig.suptitle('%d Hz'%rayF)

    fig.savefig(os.path.join(fig_dir,'rays_%dHz.png'%rayF),ldpi=300)

    plt.close('all')






