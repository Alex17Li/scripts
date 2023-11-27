import os
import numpy as np
import open3d as o3d
import pandas as pd
from datetime import datetime
import time
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
# Set the brt-devkit environment to `prod`
os.environ['BRT_ENV'] = 'prod'
# Set the directory where assets will be downloaded. brtdevkit will automatically use this directory
# to download dataset assets if a path is not passed. useful for when you have assets downloaded already
# so that assets are not downloaded again if already there
os.environ['ASSET_DIR'] = '~/.brt'  # this is the value by default
import brtdevkit
from brtdevkit.data import Dataset
from brtdevkit.core.db.athena import AthenaClient, Table
import sys, copy
sys.path.append('/home/aravindvenugopal/JupiterCVML/europa/base/src/europa')
from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import *
from aletheia_dataset_creator.config.dataset_config import * 
brtdevkit.log_level = 'debug'
from dl.utils.io_utils import normalize_image

'''
transforms_veh_base = {'front-left-left': np.array([3.2333008,0.42134647,1.45534947,-98.78456649,-1.01342691,-12.00190531]),
 'front-center-left': np.array([3.42615546,0.1,1.45488581,-98,0,-90]),
 'front-right-left': np.array([3.42889833,-0.37976413,1.45888682,-98.78456649,1.01342691,-167.9980947]),
 'rear-left': np.array([-2.03613,-0.15,3.26667,-110,0,90]),
 'side-left-left': np.array([-1.82836116,0.88797777,3.24154,-124.9990164,0,10]),
 'side-right-left': np.array([-1.53291884,-0.94007223,3.24154,-124.9990164,0,170])
}

transforms_base_cam = {'front-left-left': np.array([0,.10,0,-45,0,-45]),
 'front-center-left': np.array([0,.10,0,-45,0,-45]),
 'front-right-left': np.array([0,.10,0,-45,0,-45]),
 'rear-left': np.array([0,.15,0,-45,0,-45]),
 'side-left-left': np.array([0,.15,0,-45,0,-45]),
 'side-right-left': np.array([0,.15,0,-45,0,-45])
}
'''

'''transforms_veh_base = {'front-left-left': np.array([3.33109957, 0.40055530, 1.45711814, -0.01789753835, 0.15329543636, 1.3585903996]),
 'front-center-left': np.array([3.42615546, 0.0, 1.45488581,-0.0, 0.139626340159546, 0.0]),
 'front-right-left': np.array([3.33109957, -0.40055530, 1.45711814,0.01789753835, 0.15329543636, -1.3585903996]),
 'rear-left': np.array([-2.03613,-0.15,3.26667,-110,0,90]),
 'side-left-left': np.array([-1.82836116,0.88797777,3.24154,-124.9990164,0,10]),
 'side-right-left': np.array([-1.53291884,-0.94007223,3.24154,-124.9990164,0,170])
}

transforms_base_cam = {'front-left-left': np.array([0,.10,0,-0.785398, 0.0, 0.785398]), #roll, pitch, yaw: z, x, y ,  0.0, -0.785398, -0.785398
 'front-center-left': np.array([0,.10,0,-0.785398, 0.0, 0.785398]),
 'front-right-left': np.array([0,.10,0,-0.785398, 0.0, 0.785398]),
 'rear-left': np.array([0,.15,0,-0.785398, 0.0, 0.785398]),
 'side-left-left': np.array([0,.15,0,-0.785398, 0.0, 0.785398]),
 'side-right-left': np.array([0,.15,0,-0.785398, 0.0, 0.785398])
}'''

transforms_veh_base = {'front-left-left': np.array([3.23330080, 0.42134647, 1.45534947, -98.78456649, -1.01342691, -12.00190531]),
 'front-center-left': np.array([3.42615546, 0.1, 1.45488581, -98.0, 0.0 ,-90.0]),
 'front-right-left': np.array([3.42889833, -0.37976413, 1.45888682, -98.78456649, 1.01342691, -167.99809469]),
 'rear-left': np.array([-2.03613, -0.15, 3.26667, -110.0, 0.0, 90.0]),
 'side-left-left': np.array([-1.82836116, 0.88797777, 3.24154, -116.99901644, 0.0, 10.0]),
 'side-right-left': np.array([-1.53291884, -0.94007223, 3.24154, -116.99901644, 0.0, 170.0])
}

transforms_base_cam = {'front-left-left': np.array([0,.10,0,-0.785398, 0.0, 0.785398]), #roll, pitch, yaw: z, x, y ,  0.0, -0.785398, -0.785398
 'front-center-left': np.array([0,.10,0,-0.785398, 0.0, 0.785398]),
 'front-right-left': np.array([0,.10,0,-0.785398, 0.0, 0.785398]),
 'rear-left': np.array([0,.15,0,-0.785398, 0.0, 0.785398]),
 'side-left-left': np.array([0,.15,0,-0.785398, 0.0, 0.785398]),
 'side-right-left': np.array([0,.15,0,-0.785398, 0.0, 0.785398])
}

from numpy import sin, cos

def get_rot_mat(cloud, roll, pitch, yaw): 
    roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)
    T = np.eye(3)
    T = cloud.get_rotation_matrix_from_xyz((roll, pitch, yaw))
    return T


    
def get_transforms(cam_loc, cloud): #x, y, z
    #get rotation matrix
    params_v_b, params_b_c = transforms_veh_base[cam_loc], transforms_base_cam[cam_loc]   
    roll_vb, pitch_vb, yaw_vb = params_v_b[3], params_v_b[4], params_v_b[5] #in degrees not radians
    roll_bc, pitch_bc, yaw_bc = params_b_c[3], params_b_c[4], params_b_c[5] #in degrees not radians    
    rot_vb = get_rot_mat(cloud, roll_vb, pitch_vb, yaw_vb)               #eulerAnglesToRotationMatrix(yaw_vb, pitch_vb, roll_vb)
    rot_bc = get_rot_mat(cloud, roll_bc, pitch_bc, yaw_bc)               #eulerAnglesToRotationMatrix(yaw_bc, pitch_bc, roll_bc)
    
    t_vb = np.array([params_v_b[0], params_v_b[1], params_v_b[2]])
    t_bc = np.array([params_b_c[0], params_b_c[1], params_b_c[2]])

    vb_34, bc_34 = np.zeros((3, 4)), np.zeros((3, 4))
    vb_34[:3, :3], bc_34[:3, :3] = rot_vb.copy(), rot_bc.copy()
    vb_34[:, -1], bc_34[:, -1] = t_vb.copy(), t_bc.copy()
    return vb_34, bc_34 



def compute_transforms_all(pts_arr, cam_loc, inverse = False): #pts : Nx3
    

    pts_arr[pts_arr[:, -1] > 40] = 0
    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(pts_arr)
    
    vb, bc = get_transforms(cam_loc, pts) #3x4 matrices  
    vb_full, bc_full = np.zeros((4, 4)), np.zeros((4, 4))
    vb_full[:3], bc_full[:3]  =vb, bc
    vb_full[-1, -1], bc_full[-1, -1] = 1.0, 1.0


    #pts_inter = pts.transform(np.linalg.inv(bc_full))
    pts_final = pts.transform(np.linalg.inv(vb_full))
    return np.array(pts_final.points)