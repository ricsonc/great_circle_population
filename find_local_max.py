import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from geotiff import GeoTiff
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
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
import random
import shapely
from lib import PopCalc
from matplotlib import gridspec
import matplotlib.ticker as mticker
import pandas as pd
from ray.util.multiprocessing import Pool
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

plt.style.use('dark_background')

from spiral import compute_pops_for_pole, draw_and_compute_gc_for_pole, generate_poles

def refine_candidate(popcalc, init_lat, init_lon):
    lon_scale = np.cos(init_lat)
    delta = 0.5
    M = 21 # could make this 21 later
    
    search_lat, search_lon = np.stack(np.meshgrid(
        np.linspace(
            init_lat - delta,
            init_lat + delta,
            M
        ),
        np.linspace(
            init_lon - delta/lon_scale,
            init_lon + delta/lon_scale,
            M
        )
    ), axis=0).reshape(2, -1)

    results = [
        compute_pops_for_pole(popcalc, lat, lon)
        for lat, lon in list(zip(search_lat, search_lon))
    ]
    best_idx = np.argmax([r['total'] for r in results])
    
    return dict(
        total = results[best_idx]['total'],
        lat = search_lat[best_idx],
        lon = search_lon[best_idx],
    )

def close_to(point1, point2):
    dist = gd.geometry_length(shapely.geometry.LineString([
        point1[::-1], point2[::-1]
    ]))
    return dist < 60e3


if __name__ == '__main__':

    if False:
        gd = cartopy.geodesic.Geodesic(flattening=0)
        latitude_deg, longitude_deg = generate_poles()
        all_pops = np.load('all_pops.npy') * 25
        totals = all_pops.sum(axis=1)
        popcalc = PopCalc()

        K = 48
        # let's find the top K best candidates, ensuring they're at least 50 km apart
        idxs = all_pops.sum(axis=-1).argsort()[::-1]

        candidates = []
        for idx in idxs:
            candidate = (latitude_deg[idx], longitude_deg[idx] % 360)
            if not any((close_to(candidate, other) for other in candidates)):
                candidates.append(candidate)
                if len(candidates) >= K:
                    break

        # for candidate in candidates:
        #     _ = refine_candidate(popcalc, *candidate)

        pool = Pool(16)
        results = pool.map(
            lambda candidate: refine_candidate(popcalc, *candidate),
            candidates
        )

        best_idx = np.argmax([r['total'] for r in results])
        best_candidate = results[best_idx]

        best_lat = best_candidate['lat']
        best_lon = best_candidate['lon']

        print(best_lat, best_lon)

        st()

    else:
        draw_and_compute_gc_for_pole(
            999999,
            PopCalc(),
            48.841,
            206.765,
        )
