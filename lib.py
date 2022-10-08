from geotiff import GeoTiff
import numpy as np
from ipdb import set_trace as st
import torch
from torch import nn
from torch.nn import functional as F

class PopCalc:
    def __init__(self):
        # 30 sec -> 1km
        # we probably want ~ 5km res
        #filename = 'gpw_v4_population_density_rev11_2020_30_sec.tif'
        filename = 'gpw_v4_population_density_rev11_2020_2pt5_min.tif'
        data = GeoTiff(filename).read()
        data = np.array(data)
        data[data < 0] = 0
        data = torch.Tensor(data.astype(float))
        self.data = data
        
    def interpolate(self, points):
        points = points.copy()

        # -1,-1 is top left
        # 1, 1, is bottom right
        points[...,0] /= 180
        points[...,1] /= -90
        points = torch.Tensor(points.astype(float))

        if len(points.shape) == 2:
            points = points.unsqueeze(0).unsqueeze(0)
        elif len(points.shape) == 3:
            points = points.unsqueeze(0)
        else:
            raise Exception(f'points has shape {points.shape}, dont know what to do')

        pops = F.grid_sample(
            self.data.unsqueeze(0).unsqueeze(0),
            points,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )[0,0].numpy()

        return pops
