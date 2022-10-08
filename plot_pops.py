from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ipdb import set_trace as st
import cartopy
import numpy as np
from ipdb import set_trace as st
import sys
import cartopy.crs as ccrs
import cartopy
from cartopy import geodesic
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from ipdb import set_trace as st
import shapely.geometry as sgeom
from munch import Munch as M
from csv import reader
import sys
import time
from PIL import Image
import os
from scipy.optimize import dual_annealing, differential_evolution, basinhopping, shgo, minimize
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('TkAgg')
import random
import shapely
from lib import PopCalc
from matplotlib import gridspec
from matplotlib.scale import FuncScale

from scipy.ndimage import gaussian_filter1d

# all_pops = np.stack([np.load(f'npys/{i:06d}.npy') for i in tqdm(range(94115))])
# np.save('all_pops.npy', all_pops)
# exit()

all_pops = np.load('all_pops.npy')

# each sample point is 5km away, so mult by 25
all_pops *= 25 # this is now in units of Persons

best = all_pops.sum(axis=-1).argmax()

idxs = all_pops.sum(axis=-1).argsort()[::-1]

# plt.plot(all_pops.sum(axis=-1))
# plt.show()
st()

# find the one which lines up best with japan...

lat_from_idl = 139+180
center_idx = lat_from_idl / 360*8000
print(center_idx)

mask = all_pops[:,7030:7130].sum(axis=-1) > 2e7
foo = all_pops[mask].sum(axis=-1).argmax()
japan_idx = np.array(range(len(all_pops)))[mask][foo]
print(japan_idx)

st()

#all_pops_smoothed = gaussian_filter1d(all_pops, sigma = 8, axis=-1, mode='wrap')

# def generate_fwd_bck(offset = 200000, ratio = 0.2):
    
#     def fwd(x):
#         return np.where(x < offset, x, (x-offset)*ratio + offset)

#     def bck(y):
#         return np.where(y < offset, y, offset + (y-offset)/ratio)

#     return fwd, bck

# fwd, bck = generate_fwd_bck()

# for idx in [70734, 30000, 32000, 33000, 34000, 35000, 85000]:
#     plt.plot(all_pops_smoothed[idx])
#     plt.axhline(200000)
#     plt.yscale('function', functions=(fwd,bck))
#     plt.ylim(0,3e6)
#     plt.show()

