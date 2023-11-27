import sys, os
import pickle
import boto3, imageio, io
import matplotlib.pyplot as plt
from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import *
from aletheia_dataset_creator.config.dataset_config import * 
import open3d as o3d
import ast
import cv2
import torch
from torch.nn import functional as F
from torchvision.transforms import ColorJitter, ToTensor, Compose

from dl.utils.io_utils import normalize_image
from dl.utils.colors import classlabels_viz_cmap, classlabels_viz_norm
from dl.network.brtresnetpyramid_lite12 import BrtResnetPyramidLite12
from dl.network.models import get_logits, restore_model_from_snapshot
from dl.utils.colors import OutputType
from scipy.special import softmax 

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

def get_dataframe(ds_name, helper_path=None):
    dataset = Dataset.retrieve(name=ds_name)
    df = dataset.to_dataframe()
    return df

def get_pairs(model, df, s3, mcsv, left_id, right_id, pp_path, stopclass_names):

    egopose_data = []
    anno_pp_path, image_pp_path = pp_path[0], pp_path[1]
    anno_mcsv, image_mcsv = mcsv[0], mcsv[1]

    rowl, rowr = df[df["id"] == left_id].iloc[0], df[df["id"] == right_id].iloc[0]    
    #left_rgb = image_from_s3(rowl.artifact_debayeredrgb_0_s3_bucket, rowl.artifact_debayeredrgb_0_s3_key, s3)
    #right = image_from_s3(rowr.artifact_debayeredrgb_0_s3_bucket, rowr.artifact_debayeredrgb_0_s3_key, s3)
    #left, right = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY), cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    #egopose_data[0] = {'gps_can_data': rowl["gps_can_data"], 'gnss': rowl["gnss"], 'collected_on': rowl["collected_on"]}
    #egopose_data[1] = {'gps_can_data': rowr["gps_can_data"], 'gnss': rowr["gnss"], 'collected_on': rowr["collected_on"]}
    for files in os.listdir(image_pp_path + left_id ):
        if (files.find('stereo_output') > -1):
            stereo_output_file = files
    pp_sample = np.load(image_pp_path + left_id + "/" + stereo_output_file, allow_pickle = True)
    #rectified images
    leftr, rightr = pp_sample["left"], pp_sample["right"]
    leftr, rightr = normalize_image(leftr, hdr_mode = True), normalize_image(rightr, hdr_mode = True)
    leftr_rgb, rightr_rgb = (leftr*255).astype(np.uint8), (rightr*255).astype(np.uint8)
    leftr, rightr = cv2.cvtColor(leftr_rgb, cv2.COLOR_RGB2GRAY), cv2.cvtColor(rightr_rgb, cv2.COLOR_RGB2GRAY)
    
    calib = pp_sample["rectified_calibration_data"].tolist()  
    cloud = pp_sample["point_cloud"]
    calib["timestamp"] = np.datetime64(rowl.collected_on)

    #get seg mask
    segmask = None
    if(anno_pp_path is not None and os.path.exists(anno_pp_path + left_id)):
        for i in os.listdir(anno_pp_path + left_id):
            if os.path.isfile(os.path.join(anno_pp_path + left_id,i)) and 'rectification_output' in i:
                segmask = np.load(anno_pp_path + left_id + "/" + i, allow_pickle = True)["left"]
                #get human label
                #print(left_id)
                
                labelmap = anno_mcsv[anno_mcsv["id"] == left_id]["label_map"].values[0]
                stopclass_labels = {}
                labelmap = ast.literal_eval(labelmap)
                for k, v in labelmap.items():
                    if(v in stopclass_names):
                        stopclass_labels[v] = int(k)
    else:
        #predicted seg mask
        output = segment(model, leftr_rgb.copy(), cloud[:, :, 2].copy(), max_depth=100)
        segmask = np.expand_dims(np.argmax(output, axis=-1).astype(np.uint8), axis = -1)
        confidence = softmax(output, axis=2)
        pred_conf = np.amax(confidence, axis=-1)
        #print(pred_conf.min(), pred_conf.max())
        low_pred_idxes = np.argwhere(pred_conf < 0.6)
        segmask[low_pred_idxes[:, 0], low_pred_idxes[:, 1]] = 0 
        stopclass_labels = {'Humans': 5, "Tractors or Vehicles": 6}
        #plt.imshow(leftr_rgb.copy())
        #plt.show()
        #plt.imshow(segmask)
        #plt.show()

        #plt.close()


    #return left, right, left_rgb, leftr, rightr, calib, leftr_rgb, segmask, cloud, stopclass_labels
    return leftr_rgb, rightr_rgb, segmask, cloud, stopclass_labels, calib, egopose_data #calib also contains timestamp

def create_model(model_name, model_path):
    num_classes = 7
    input_dims = 4
    output_type = OutputType.MULTISCALE
    upsample_mode = 'nearest'
    seg_model = get_logits(model_name=model_name,
                           num_classes=num_classes,
                           input_dims=input_dims,
                           output_type=output_type,
                           upsample_mode=upsample_mode,
                          )

    restore_model_from_snapshot(model_path, seg_model, optimizer=None)
    seg_model.eval()
    return seg_model
    
def segment(model, rgb_left, depth, max_depth):
    depth = depth.copy() / max_depth
    depth[depth > 1.0] = 1.0
    depth[depth < 0.0] = 0.0
    
    model_input = rgb_left.astype(np.float32) / 255.0
    model_input = np.append(model_input, np.expand_dims(depth, -1), axis=2)
    model_input = np.expand_dims(model_input.transpose((2, 0, 1)), 0)
    with torch.no_grad():
        output = model(torch.tensor(model_input))[0].detach().numpy()[0]
        output = output.transpose((1, 2, 0))
    return output

# FUNCTIONS    
def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))

def get_floor_plane(pcd, dist_threshold=0.02, visualize=False):
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model    
    return plane_model


def image_from_s3(bucket, key, s3):
    bucket = s3.Bucket(bucket)
    image = bucket.Object(key)
    img_data = image.get().get('Body').read()
    return imageio.imread(io.BytesIO(img_data))

def get_groups(df, keys, halo = False, halo_mcsv = None): #returns all data including image IDs for each group ID, BUT SEARCHES ONLY IN THIS DATASET AND NOT ALL OF ALETHEIA 
    group_ids = df[["group_id", "collected_on"]].drop_duplicates().sort_values(by=['collected_on'])["group_id"].drop_duplicates().tolist()
    lens = []
    groups = []
    j = 0
    '''if(not halo):
        loc_dict = {"front-right-left": 0, "front-right-right": 1, "front-left-left": 2, "front-left-right":3, "front-center-left": 4, 
                    "front-center-right": 5, "rear-left": 6, "rear-right":7, "side-right-left":  8, "side-right-right": 9, "side-left-left": 10, 
                    "side-left-right": 11  }
    else:
        loc_dict =  {'T01': 0, 'T02': 1, 'T03': 2,'T04': 4,
       'T05': 5, 'T06': 6, 'T07': 7, 'T08': 8,
       'T09': 9, 'T10': 10, 'T11': 11,'T12': 12,
       'T13': 13, 'T14': 14, 'T15': 15, 'T16': 16}'''
    print("starting loop")
    if(not halo):  #unoptimized for debugging
        for gid in group_ids:
            j += 1
            if(j%10001==0):
                print(j, len(groups))
            loc_ids = {}
            all_extrinsics = {}
            group  = df[df["group_id"] == gid]
            error_flag = False
            all_ids_present = True
            for key in keys:    
                #Error check
                if(key[0] == "T" and halo == False):
                    error_flag = True
                assert error_flag == False, "Cam locations are Halo locations but config is not Halo"

                val = group[group["camera_location"]==key]
                ext = None 
                if not val.empty:
                    #ext = val["calibration_data"].tolist()[0]#["p1"]
                    val = val["id"].values[0]
                else:
                    val = None    
                    all_ids_present = False
                    break

                loc_ids[key] = val 
                #all_extrinsics[key] = ext
            if(all_ids_present == False):
                continue
            loc_ids["length"] = len(group)
            loc_ids["group_id"] = gid
            loc_ids["ts_first"] = np.array(df[df["group_id"] == gid]["collected_on"]).astype('datetime64').min()
            loc_ids["ts_last"] = np.array(df[df["group_id"] == gid]["collected_on"]).astype('datetime64').max()
            #loc_ids["params"] = all_extrinsics
            groups.append(loc_ids)
            #if(len(groups) == 1000):
            #return groups, group_ids
    else:
        if(halo_mcsv is not None):
            mcsv_group_ids =  halo_mcsv[["group_id", "collected_on"]].drop_duplicates().sort_values(by=['collected_on'])["group_id"].drop_duplicates().tolist()
        for gid in group_ids:
            j += 1

            loc_ids = {}
            #all_transforms = {}
            group  = df[df["group_id"] == gid]
            error_flag = False
            
            for key in keys:    #get group ID, all image IDs, group length and timestamps
                
                #Error check
                if(key[0] == "T" and halo == False):
                    error_flag = True
                assert error_flag == False, "Cam locations are Halo locations but config is not Halo"

                val = group[group["camera_location"]==key]
                #ext = None 
                if not val.empty:
                    #rot = halo_mcsv[halo_mcsv['id'] == val]
                    #print(rot)
                    #ext = 0 #val["calibration_data"].tolist()[0]#["p1"]
                    val = val["id"].values[0]
                    #print(val)
                    #print(halo_mcsv[halo_mcsv['id'] == val]["online_calibration_results"])
                    #rot = halo_mcsv[halo_mcsv['id'] == val]["online_calibration_results"].values[0]
                    #print(rot)
                else:
                    val = None    

                loc_ids[key] = val 
                #all_transforms[key] = ext
 
            loc_ids["length"] = len(group)
            loc_ids["group_id"] = gid
            loc_ids["ts_first"] = np.array(df[df["group_id"] == gid]["collected_on"]).astype('datetime64').min()
            loc_ids["ts_last"] = np.array(df[df["group_id"] == gid]["collected_on"]).astype('datetime64').max()

            
  
            pair_dicts = [HALO_CENTER_CAMERA_PAIRS, HALO_NON_CENTER_CAMERA_PAIRS]
            all_left_keys = set().union(*pair_dicts)
            #get extrinsics 
            all_transforms = {}
            for key in keys:
                key_ext_dict = {}
            
                #print(key)
                image_id = loc_ids[key]
                #print(image_id)
                mcsv_rows = halo_mcsv[halo_mcsv['id'] == image_id][['camera_location', 'camera_location_right', 'online_calibration_results']]
                row_list = list(mcsv_rows.itertuples(index=False, name=None))
                if(len(mcsv_rows) == 0):
                    all_transforms[key] = None
                    continue
                for item in row_list:
                    key_ext_dict[item[1]] = ast.literal_eval(item[2])

                all_transforms[key] = key_ext_dict.copy()    
            
            loc_ids["transforms"] = all_transforms

            groups.append(loc_ids)

            
            
    return groups, group_ids

def store_data_single_frame(df, groups, pp_path, save_path, s3, full_required = False, save=True, halo=False):
    count = 0 
    for item_id, item in enumerate(groups):

        if(full_required):
            if(item["length"] != 12):
                continue
            else:
                count += 1
        #count += 1
        if(save and not os.path.exists(save_path + str(count))):
            os.mkdir(save_path + str(count))
        metadata_dict = item.copy()
        with open(save_path + str(count) +"/"+ 'metadata_dict.pkl', 'wb') as f:
                pickle.dump(metadata_dict, f)
        if(not halo):
            camera_locs = list(item.keys())[:12]
        else:
             camera_locs = list(item.keys())[:16]  

        print(item_id, item["length"], item["group_id"])

        for i, loc in enumerate(camera_locs):
                            
            id =  item[loc]
            if id is not None:
                row = df[df["id"] == id].iloc[0]
                im = image_from_s3(row.artifact_debayeredrgb_0_s3_bucket, row.artifact_debayeredrgb_0_s3_key, s3)
                np.save(save_path + str(count)+"/"+loc+"_image_original", im)
                if(not halo):
                    if(i%2==0):
                        pp_sample = np.load(pp_path + id + "/stereo_output.npz", allow_pickle = True)
                        #rectified images
                        left_im, right_im, point_cloud = pp_sample["left"], pp_sample["right"], pp_sample["point_cloud"]
                        calib_data = pp_sample["rectified_calibration_data"]
                        left_im, right_im = normalize_image(left_im, hdr_mode = True), normalize_image(right_im, hdr_mode = True)
                        left_im, right_im = (left_im*255).astype(np.uint8), (right_im*255).astype(np.uint8)
                        if(save):
                            #print("Saving ... ")
                            np.save(save_path + str(count)+"/"+loc+"_image_rect", normalize_image(left_im, hdr_mode = True))
                            np.save(save_path + str(count)+"/"+loc[:-4]+"right_image_rect", normalize_image(right_im, hdr_mode = True))
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))
                            colors = left_im.copy().reshape(-1, 3) #cv2.cvtColor(segmask[:, :, 0]*50,cv2.COLOR_GRAY2RGB) 
                            pcd.colors = o3d.utility.Vector3dVector(colors)               
                            o3d.io.write_point_cloud(save_path + str(count)+"/"+loc+"_cloud.pcd", pcd)
                            np.save(save_path + str(count)+"/"+loc+"_calib_data" , calib_data)      
                            #np.save(save_path + str(count)+"/"+loc+"_cloud", point_cloud)
                else:
                    if(loc in HALO_CAMERA_PAIRS.keys()):
                        left_loc, right_loc = loc, HALO_CAMERA_PAIRS[loc]
                        pp_sample = np.load(pp_path + id + "/stereo_output.npz", allow_pickle = True)
                        #rectified images
                        left_im, right_im, point_cloud = pp_sample["left"], pp_sample["right"], pp_sample["point_cloud"]
                        calib_data = pp_sample["rectified_calibration_data"]
                        left_im, right_im = normalize_image(left_im, hdr_mode = True), normalize_image(right_im, hdr_mode = True)
                        left_im, right_im = (left_im*255).astype(np.uint8), (right_im*255).astype(np.uint8)
                        if(save):
                            #print("Saving ... ")
                            np.save(save_path + str(count)+"/"+left_loc+"_image_rect", normalize_image(left_im, hdr_mode = True))
                            np.save(save_path + str(count)+"/"+right_loc+"_image_rect", normalize_image(right_im, hdr_mode = True))
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))
                            colors = left_im.copy().reshape(-1, 3) #cv2.cvtColor(segmask[:, :, 0]*50,cv2.COLOR_GRAY2RGB) 
                            pcd.colors = o3d.utility.Vector3dVector(colors)               
                            o3d.io.write_point_cloud(save_path + str(count)+"/"+left_loc+"_cloud.pcd", pcd)
                            np.save(save_path + str(count)+"/"+left_loc+"_calib_data" , calib_data)  
                            #np.save(save_path + str(count)+"/"+loc+"_cloud", point_cloud)                        
                    #todo once I get to know the structure of pp output
        if(count == 10):
            break

def store_data_sequence(df, seq, pp_path, save_path, s3, count, save=True, halo = False): #sequence is a sub-list of groups
    if(save and not os.path.exists(save_path + str(count))):
        os.mkdir(save_path + str(count))
        os.mkdir(save_path + str(count)+"/original")
        os.mkdir(save_path + str(count)+"/rectified")
        os.mkdir(save_path + str(count)+"/pointcloud")
        os.mkdir(save_path + str(count)+"/calibration")
    print("Length of sequence", len(seq))
    metadata_dicts = seq.copy()
    with open(save_path + str(count) +"/"+ 'metadata_dicts.pkl', 'wb') as f:
            pickle.dump(metadata_dicts, f)
    for item_id, item in enumerate(seq): 
        print(item_id, item["length"], item["group_id"])
        if(not halo):
            camera_locs = list(item.keys())[:12]
        else:
            camera_locs = list(item.keys())[:16]           
        for i, loc in enumerate(camera_locs):                
            id =  item[loc]
            if id is not None:
                row = df[df["id"] == id].iloc[0] 
                im = image_from_s3(row.artifact_debayeredrgb_0_s3_bucket, row.artifact_debayeredrgb_0_s3_key, s3)
                np.save(save_path + str(count)+"/original/"+loc+"_image_original_"+str(item_id), im)
                if(not halo):
                    if(i%2==0):

                        pp_sample = np.load(pp_path + id + "/stereo_output.npz", allow_pickle = True)
                        #rectified images
                        left_im, right_im, point_cloud = pp_sample["left"], pp_sample["right"], pp_sample["point_cloud"]
                        left_im, right_im = normalize_image(left_im, hdr_mode = True), normalize_image(right_im, hdr_mode = True)
                        left_im, right_im = (left_im*255).astype(np.uint8), (right_im*255).astype(np.uint8)
                        calib_data = pp_sample["rectified_calibration_data"]
                        if(save):
                            np.save(save_path + str(count)+"/rectified/"+loc+"_image_rect_"+str(item_id), normalize_image(left_im, hdr_mode = True))
                            if(loc == "front-center-left"):
                                plt.imshow(left_im)
                                plt.show()
                            np.save(save_path + str(count)+"/rectified/"+loc[:-4]+"right_image_rect_"+str(item_id), 
                                    normalize_image(right_im, hdr_mode = True))                                 
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))   
                            colors = left_im.copy().reshape(-1, 3) #cv2.cvtColor(segmask[:, :, 0]*50,cv2.COLOR_GRAY2RGB) 
                            pcd.colors = o3d.utility.Vector3dVector(colors)      
                            o3d.io.write_point_cloud(save_path + str(count)+"/pointcloud/"+loc+"_cloud_"+str(item_id)+".pcd", pcd)
                            #np.save(save_path + str(count)+"/pointcloud/"+loc+"_cloud_"+str(item_id), point_cloud)
                            np.save(save_path + str(count)+"/calibration/"+loc+"_calib_data_"+str(item_id) , calib_data)
                else:
                    if(loc in HALO_CAMERA_PAIRS.keys()):
                        left_loc, right_loc = loc, HALO_CAMERA_PAIRS[loc]
                        pp_sample = np.load(pp_path + id + "/stereo_output.npz", allow_pickle = True)
                        #rectified images
                        left_im, right_im, point_cloud = pp_sample["left"], pp_sample["right"], pp_sample["point_cloud"]
                        calib_data = pp_sample["rectified_calibration_data"]
                        left_im, right_im = normalize_image(left_im, hdr_mode = True), normalize_image(right_im, hdr_mode = True)
                        left_im, right_im = (left_im*255).astype(np.uint8), (right_im*255).astype(np.uint8)
                        if(save):
                            #print("Saving ... ")
                            np.save(save_path + str(count)+"/"+left_loc+"_image_rect", normalize_image(left_im, hdr_mode = True))
                            np.save(save_path + str(count)+"/"+right_loc+"_image_rect", normalize_image(right_im, hdr_mode = True))
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))
                            colors = left_im.copy().reshape(-1, 3) #cv2.cvtColor(segmask[:, :, 0]*50,cv2.COLOR_GRAY2RGB) 
                            pcd.colors = o3d.utility.Vector3dVector(colors)               
                            o3d.io.write_point_cloud(save_path + str(count)+"/"+left_loc+"_cloud.pcd", pcd)
                            np.save(save_path + str(count)+"/"+left_loc+"_calib_data" , calib_data)  
                            #np.save(save_path + str(count)+"/"+loc+"_cloud", point_cloud)                        
                    #todo once I get to know the structure of pp output               
                        