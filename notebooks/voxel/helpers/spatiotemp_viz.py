import sys, os
import numpy as np
import open3d as o3d
import pickle
import boto3, imageio, io
import matplotlib.pyplot as plt
from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import *
from aletheia_dataset_creator.config.dataset_config import * 
from dl.utils.io_utils import normalize_image
from dl.utils.image_transforms import preprocess_pointcloud

HALO_CENTER_CAMERA_PAIRS = {"T02": "T03", "T06": "T07", "T10": "T11", "T14": "T15"}
HALO_NON_CENTER_CAMERA_PAIRS = {
    "T01": "T03",
    "T02": "T04",
    "T05": "T07",
    "T06": "T08",
    "T09": "T11",
    "T10": "T12",
    "T13": "T15",
    "T14": "T16",
}


def plot_cloud(point_cloud, n_left_im, loc=None):

    if(loc is None):
        loc = "default"
    #point_cloud[point_cloud[:, -1] > 40] = np.zeros(3)
    #x_mean = point_cloud[:, 0].mean()
    #y_mean = point_cloud[:, 1].mean()
    #point_cloud[point_cloud[:, 1] < y_mean] = 0
    # Create PointCloud class
    '''pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    print(n_left_im.shape)
    colors = n_left_im.reshape(-1, 3) #cv2.cvtColor(segmask[:, :, 0]*50,cv2.COLOR_GRAY2RGB) 
    pc.colors = o3d.utility.Vector3dVector(colors)    
    o3d.visualization.draw_geometries([pc], window_name=loc,
              zoom=0.3412,
              front=[0.4257, -0.2125, -0.8795],
              lookat=[2.6172, 2.0475, 1.532],
              up=[-0.0694, -0.9768, 0.2024])'''

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud.copy())
    colors = n_left_im
    pc.colors = o3d.utility.Vector3dVector(colors)    

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pc)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()
    
def visualize_data(item, keys, pp_path, viz = True, halo = False): #item is the group dict
        
        left_core_locs = {"front-right-left", "front-left-left", "front-center-left", "rear-left", "side-right-left",  "side-left-left"}
        all_lefts, all_clouds = {}, {}
        if(not halo):
            camera_locs = left_core_locs #list(item.keys())[:12]
        else:
            camera_locs = list(item.keys())[:16] 
        #print("camera locs are", camera_locs)
        for i, loc in enumerate(keys):
            id =  item[loc]
            if(id is not None):
                if(not halo):
                    if(loc in left_core_locs): #looking at the left camera locs
                        pair = loc[:-5]
                        left_id, right_id = item[pair+"-left"], item[pair+"-right"]                  
                        for files in os.listdir(pp_path + left_id):
                            if (files.find('stereo_output') > -1):
                                stereo_output_file = files
                        pp_sample = np.load(pp_path + id + "/" + stereo_output_file, allow_pickle = True)
                        left_im, right_im, point_cloud = pp_sample["left"], pp_sample["right"], pp_sample["point_cloud"]
                        depth = point_cloud[:, :, -1]
                        point_cloud = point_cloud.reshape(-1, 3)
                        calib_data = pp_sample["rectified_calibration_data"]
                        idxes = np.where(point_cloud[:, -1] < 40)[0]
                        point_cloud[point_cloud[:, -1] > 40] = 0.0     #Filtering these points
                        n_left_im = normalize_image(left_im, hdr_mode = True)
                        n_left_im_flat = n_left_im.reshape(-1, 3)[idxes]
                        #print("camera location:", loc)
                        
                        
                        if(viz):
                            
                            fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 10))
                            ax1.set_title("Left Depth")  
                            ax1.imshow(depth)
                            ax2.set_title("Left Rectified")   
                            ax2.imshow(n_left_im)   
                            fig.show()
                            plt.show()
                            plt.close()
    
                            
                            plot_cloud(point_cloud, n_left_im.reshape(-1, 3), loc)
                        
                        all_clouds[loc] = point_cloud
                        all_lefts[loc] =  n_left_im.reshape(-1, 3)
                else:

                    if(loc in item['transforms'].keys()):

                        left_loc = loc
                        
                        stereo_output_files = []
                        if(os.path.exists(pp_path + id)):
                            for files in os.listdir(pp_path + id):
                                if (files.find('stereo_output') > -1):
                                    stereo_output_files.append(files)       

                        if(stereo_output_files is not None):
                            for file in stereo_output_files:
                                    right_loc = str(file).split('.')[0].split('_')[-1]

                                    pp_sample = np.load(pp_path + id + "/" + file, allow_pickle = True)
                                    left_im, right_im, point_cloud = pp_sample["left"], pp_sample["right"], pp_sample["point_cloud"]
                                    rot = pp_sample["rectified_calibration_data"].tolist()['R1']
                                    rot = np.array(rot).reshape(3, 3)
                                    rot_inv = np.linalg.inv(rot)
                                    point_cloud = point_cloud.reshape(-1, 3)
                                    point_cloud = (rot_inv @ point_cloud.T).T
                                    print(point_cloud.shape)
                                    n_left_im = normalize_image(left_im, hdr_mode = True)
                                    n_right_im = normalize_image(right_im, hdr_mode = True)
                                    print("camera locations:", left_loc, right_loc)
                                    if(viz):

                                        plt.title("Left Rectified"+ str(left_loc))  
                                        plt.imshow(n_left_im)   
                                        plt.show()

                                        plt.title("Right Rectified"+ str(right_loc))   
                                        plt.imshow(n_right_im)   
                                        plt.show()
                                        plt.close()


                                        plot_cloud(point_cloud, n_left_im.reshape(-1, 3), loc)

                                    all_clouds[left_loc + "_" + right_loc] = point_cloud
                                    all_lefts[left_loc + "_" + right_loc] =  n_left_im.reshape(-1, 3)      
         
        return all_lefts, all_clouds