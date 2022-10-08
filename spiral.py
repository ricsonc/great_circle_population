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

def create_fig_and_axes():
    # just in case
    plt.switch_backend('agg')
    
    plt.close() # long story...
    #globe = cartopy.crs.Globe(flattening=0)
    projs = [
        ccrs.Orthographic(central_latitude=20, central_longitude=-80),
        ccrs.Orthographic(central_latitude=30, central_longitude=70),
    ]
    fig = plt.figure(figsize=(24,16))
    gs = gridspec.GridSpec(2, 2, height_ratios = [7, 2], width_ratios=[1,1], hspace=0.2, wspace=0.05)
    pop_ax = plt.subplot(gs[1,:], axes_class=HostAxes)
    axs = [
        plt.subplot(gs[0,0], projection = projs[0]),
        plt.subplot(gs[0,1], projection = projs[1]),
    ]

    for side in ['right', 'left', 'bottom', 'top']:
        pop_ax.axis[side].set_visible(False)
    
    twins = [ParasiteAxes(pop_ax, sharey=pop_ax) for _ in range(2)]
    for i, twin in enumerate(twins):
        pop_ax.parasites.append(twin)
        twin.axis['right'].set_visible(False)
        twin.axis['left'].set_visible(False)
        twin.axis['bottom'].set_visible(False)

        # i do not understand this at all
        if i:
            twin.axis['top'].set_visible(False)
        else:
            twin.axis['top'].set_visible(True)
            twin.axis['top'].line.set_color('white')
        
        twin.xaxis.set_ticks_position('top')
        twin.set_yticks([])
        twin.set_xticks([])        
        twin.get_xaxis().set_visible(True)

    pop_ax.set_xticks([])
    
    for ax in axs + [pop_ax]:
        ax.axis('off')
    for ax in axs + [pop_ax] + twins:
        ax.patch.set_visible(False)

    for ax in [pop_ax] + twins:
        ax.set_xlim(0, 1)

    return fig, axs, pop_ax, twins

def generate_poles(freq = 200):
    # resolution = 10000 km / freq
    # 100km width, 50 km stepping, so 200 rots
    ts = [0.0]

    # 1.7 was chosen mostly by visual inspection
    rate = 1.7 / freq
    while ts[-1] < 1:
        t = ts[-1]
        vel = ((-np.pi/2)**2 + (2*np.pi*freq)**2 * np.cos(np.pi/2*(1-t))**2)**0.5
        ts.append(t + rate/vel)

    ts = np.array(ts)

    latitude = np.pi/2 * (1-ts)
    longitude = ts * freq * 2 * np.pi

    latitude_deg = 180/np.pi * latitude
    longitude_deg = 180/np.pi * longitude

    return latitude_deg, longitude_deg

def draw_background(ax, figure):
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=1,
        color='lime',
        alpha=1.0,
        linestyle='dotted',
        clip_on=False,
        figure=figure,
    )

    gl.ylocator = mticker.FixedLocator(np.linspace(-90., 90., 19))
    gl.xlocator = mticker.FixedLocator(np.linspace(-180., 170., 36))

    res = '110m'
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', res, edgecolor='red', facecolor='none'), linewidth=0.5)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', res, edgecolor='red', facecolor='none'), linewidth=0.5)

    # draw points

    filename = 'gpw_v4_population_density_rev11_2020_2pt5_min.tif'
    data = GeoTiff(filename).read()
    data = np.array(data)
    data[data < 0] = 0
    data_rgb = plt.cm.inferno(plt.Normalize(0, 21)(data**0.4))[...,:3]
    ax.imshow(
        data_rgb,
        extent = [-180, 180, -90, 90],
        transform = ccrs.PlateCarree(),
    )
    # cities = pd.read_csv('worldcities_parsed.csv')
    # ax.scatter(
    #     cities.lon,
    #     cities.lat,
    #     transform = ccrs.PlateCarree(),
    #     s=cities.population**0.5/500,
    #     c='violet',
    #     alpha=np.minimum(1, cities.population.values**0.5/2000)
    # )
    

    ax.set_global()

def draw_and_save_bkg():
    fig, axs, pop_ax, twins = create_fig_and_axes()
    for ax in axs:
        draw_background(ax, fig)
    bkg_path = 'bkg.png'
    plt.savefig(bkg_path, dpi=100, pad_inches=0.5, bbox_inches='tight', transparent=False)
    plt.close()

def find_nearest_point_in_path(path, targets, mask = None):
    N = len(path)
    # (x-y)^2 = x^2 + y^2 - 2xy
    x2 = (path**2).reshape(-1, 1)
    y2 = (targets**2).reshape(1,-1)
    xy = path.reshape(-1,1) * targets.reshape(1,-1)
    dists_sq = x2+y2-2*xy
    if not isinstance(mask, type(None)):
        dists_sq[mask] = float('inf')
    lowest = np.argmin(dists_sq, axis=0)
    return lowest, dists_sq.min(axis=0)

def compute_pops_for_pole(popcalc, lat, lon):
    gd = cartopy.geodesic.Geodesic(flattening=0)
    dense_points = gd.circle(
        lon=lon, 
        lat=lat,
        radius = 6378137.0 * np.pi/2,
        n_samples=8000,
        endpoint=True
    )
    lons = dense_points[:,0]
    offset = np.where(lons[1:] < lons[:-1])[0][0]+1
    dense_points = np.roll(dense_points, -offset, axis=0)


    delta = 5e3 # 5km
    sample_count = 10 #50 km each

    t0 = time.time()
    endpoint = [(lon, lat)]
    _, azimuths, _ = gd.inverse(dense_points, endpoint).T

    sample_pointss = []
    for j in range(-sample_count, sample_count+1):
        sample_pointss.append(gd.direct(dense_points, azimuths, delta * j))

    sample_grid = np.stack(sample_pointss, axis=0)[...,:2]

    # plotting code here
    pops = popcalc.interpolate(sample_grid) * 25
    # 25 because of our spacing
    
    # technically sigma=8 is more correct
    smoothed_pops = gaussian_filter1d(pops.sum(axis=0), sigma = 20, axis=-1, mode='wrap')

    return dict(
        dense_points = dense_points,
        pops = pops,
        smoothed_pops = smoothed_pops,
        total = pops.sum(),
    )
        
def draw_and_compute_gc_for_pole(i, popcalc, lat, lon):

    fig, axs, pop_ax, twins = create_fig_and_axes()
    gd = cartopy.geodesic.Geodesic(flattening=0)
    plot_pointss = [
        gd.circle(
            lon=lon,
            lat=lat,
            radius = 6378137.0 * np.pi/2 + delta,
            n_samples=240,
            endpoint=True
        )
        for delta in [-5e4, 5e4]
    ]

    ts = np.linspace(0, 1, 8000)

    popcalc_result = compute_pops_for_pole(popcalc, lat, lon)
    
    
    pop_ax.plot(ts, popcalc_result['smoothed_pops'], c='violet')
    pop_ax.set_ylim(-1e4, 14e5)

    lon_ticks = np.linspace(-175.0, 180.0, 72)
    lat_ticks = np.linspace(-90.0, 90.0, 37)

    dense_points = popcalc_result['dense_points']
    lon_tick_pos, lon_errors = find_nearest_point_in_path(dense_points[:,0], lon_ticks)

    lats = dense_points[:,1]
    inc_mask = lats > np.roll(lats, 1)
    dec_mask = lats < np.roll(lats, 1)    
    lat_tick_pos_inc, lat_errors_inc = find_nearest_point_in_path(lats, lat_ticks, inc_mask)
    lat_tick_pos_dec, lat_errors_dec = find_nearest_point_in_path(lats, lat_ticks, dec_mask)
    
    lat_tick_pos = np.concatenate([lat_tick_pos_inc, lat_tick_pos_dec])
    lat_errors = np.concatenate([lat_errors_inc, lat_errors_dec])
    lat_ticks = np.concatenate([lat_ticks]*2)

    lon_tick_pos = lon_tick_pos/8000
    lat_tick_pos = lat_tick_pos/8000

    lon_mask = (lon_errors < 0.1) & (np.abs(lon_tick_pos - 0.5) < 0.48)
    lat_mask = (lat_errors < 0.1) & (np.abs(lat_tick_pos - 0.5) < 0.48)

    lon_tick_pos = list(lon_tick_pos[lon_mask])
    lon_tick_labels = [str(x) if not(x % 30) else '' for x in lon_ticks[lon_mask].astype(int)]

    lat_tick_pos = list(lat_tick_pos[lat_mask])
    lat_tick_labels = [str(x) if not (x % 10) else '' for x in lat_ticks[lat_mask].astype(int)]

    twins[0].set_xticks(lat_tick_pos, lat_tick_labels)
    twins[1].set_xticks(lon_tick_pos, lon_tick_labels)
    
    twins[0].tick_params(axis='x', which='both', length=10, direction='in', pad=-25, colors='white')
    twins[1].tick_params(axis='x', which='both', length=10, direction='out', pad=0, colors='white')
    
    for points in plot_pointss:
        points[:,0] = np.unwrap(points[:,0], period = 360.0)

    geoms = [
        shapely.geometry.LineString(points)
        for points in plot_pointss
    ]

    for ax in axs:
        ax.add_geometries(geoms, crs=cartopy.crs.PlateCarree(), facecolor='none', edgecolor='cyan', linewidth=0.8)

    plt.figtext(
        0.525,
        0.35,
        f'{int(popcalc_result["total"]//1e6)}M',
        fontsize='xx-large',
        horizontalalignment='right',
        fontfamily='monospace',
        color='white',
    )

    out_fn = f'out/{i:06d}.png'
    npy_fn = f'npys/{i:06d}.npy'
    
    plt.savefig(out_fn, dpi=100, pad_inches=0.5,bbox_inches='tight', transparent=True)

    np.save(npy_fn, popcalc_result['pops'].sum(axis=0)) # collapse one axis
    
    background = Image.open('bkg.png')
    with open(out_fn, 'rb') as f:        
        img = Image.open(f)
        img = Image.alpha_composite(background, img)

    #oh... WHITE
    with open(out_fn, 'wb') as f:
        img.save(f)

if __name__ == '__main__':

    #draw_and_save_bkg()
    
    latitude_deg, longitude_deg = generate_poles()
    N = len(latitude_deg)

    popcalc = PopCalc()
    
    # pool = Pool(16)
    # pool.map(
    #     lambda i: draw_and_compute_gc_for_pole(i, popcalc, latitude_deg[i], longitude_deg[i]),
    #     range(0, N)
    # )

    for i in tqdm(range(10)):    
        draw_and_compute_gc_for_pole(i, popcalc, latitude_deg[i], longitude_deg[i])

    # best is 23307
