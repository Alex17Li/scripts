import os, copy
import numpy as np
import open3d as o3d
import math
import matplotlib.pyplot as plt

baseline_front = 0.2 #20 cm
baseline_rearside = 0.3
#DEFINE ALL TRANSFORMATION MATRICES FOR CORE DATA



T1c, T2c, T3c = np.eye(4), np.eye(4), np.eye(4)
T4c, T5c, T6c = np.eye(4), np.eye(4), np.eye(4)

R1c = [ 0.977988, -0.207912, 0.017687,-0.014660, -0.153016, -0.988115,0.208147, 0.966105, -0.152696]
R2c = [ 0.000000, -1.000000, 0.000000,-0.139173, -0.000000, -0.990268,0.990268, 0.000000, -0.139173]
R3c = [ -0.977988, -0.207912, -0.017687,-0.014660, 0.153016, -0.988115,0.208147, -0.966105, -0.152696]
R4c = [0.984808, 0.173648, 0.000000,0.099598, -0.564849, -0.819162,-0.142246, 0.806717, -0.573562]
R5c = [ 0.000000, 1.000000, 0.000000, 0.342020, -0.000000, -0.939693, -0.939693, 0.000000, -0.342020]
R6c =  [ -0.984808, 0.173648, 0.000000,0.099598, 0.564849, -0.819162,-0.142246, -0.806717, -0.573562]


T1c[:3, :3], T2c[:3, :3], T3c[:3, :3] = np.array(R1c).reshape(3, 3), np.array(R2c).reshape(3, 3), np.array(R3c).reshape(3, 3)
T4c[:3, :3], T5c[:3, :3], T6c[:3, :3] = np.array(R4c).reshape(3, 3), np.array(R5c).reshape(3, 3), np.array(R6c).reshape(3, 3)

T1c[:-1, 3], T2c[:-1, 3], T3c[:-1, 3] = [-3.100266, 1.549926, -0.857841], [0.100000, 1.917556, -3.190331], [3.300266, 1.549926, -0.857841]
T4c[:-1, 3], T5c[:-1, 3], T6c[:-1, 3] = [1.646389, 3.339020, 0.882802], [0.150000, 3.766063, -0.796069], [-1.346389, 3.339020, 0.882802]


l2r_front = np.eye(4)
l2r_front[0, -1] = -1*baseline_front

l2r_rs = np.eye(4)
l2r_rs[0, -1] = -1*baseline_rearside

#These are for ego to left cam and ego to left cam => left cam to right cam
transformsc = {"front-left-left": T1c,
               "front-left-right": l2r_front@T1c,
            "front-center-left": T2c,
               "front-center-right": l2r_front@T2c,
            "front-right-left": T3c,
               "front-right-right": l2r_front@T3c,
            "side-right-left": T6c,
               "side-right-right": l2r_rs@T6c,
            "rear-left": T5c,
               "rear-right": l2r_rs@T5c,
            "side-left-left": T4c,
               "side-left-right": l2r_rs@T4c
              }


#DEFINE ALL TRANSFORMATION MATRICES FOR HALO DATA



def rotmat(r, p, y):
    roll_rot = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    pitch_rot = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    yaw_rot = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    rot_tf = np.matmul(yaw_rot, np.matmul(pitch_rot, roll_rot))
    return rot_tf.T


#RPY
rot = {'T01': [-110.0, 0.0, -45.0], 'T02': [-110.0, 0.0, -105], 'T03': [-110.0, 0.0, 45.0],'T04': [-110.0, 0.0, -15.0],
       #'T01': [-1.91986, 0.0, -2.35619], 'T02': [-1.91986, 0.0, -7.59218], 'T03': [-1.91986, 0.0, -1.8326],'T04': [-1.91986, 0.0, -7.06858],
       'T05': [-1.91986, 0.0, -3.92699], 'T06': [-1.91986, 0.0, -2.87979], 'T07': [-1.91986, 0.0, -3.40339], 'T08': [-1.91986, 0.0, -2.35619],
       'T09': [1.91986, 0.0, -5.49779], 'T10': [-1.91986, 0.0, -4.45059], 'T11': [-1.91986, 0.0, -4.97419],'T12': [-1.91986, 0.0, -3.92699],
       'T13': [-1.91986, 0.0, -7.06858], 'T14': [-1.91986, 0.0, -6.02139], 'T15': [-1.91986, 0.0, -6.54498], 'T16': [-1.91986, 0.0, -5.49779]}
#XYZ
tran = {'T01': [0.15, 0.355, 3.637], 'T02': [0.125, 0.255, 3.637], 'T03':[0.125, -0.255, 3.637],'T04':[0.15, -0.355, 3.637],
        'T05':[-0.875, -1.155, 3.657], 'T06':[-0.97, -1.13, 3.657], 'T07':[-1.33, -1.13, 3.677], 'T08':[-1.425, -1.155, 3.677],
        'T09':[-2.075, -0.355, 3.647], 'T10':[-2.05, -0.255, 3.647], 'T11':[-2.05, 0.255, 3.647],'T12':[-2.075, 0.355, 3.647],
        'T13':[-1.425, 1.155, 3.677], 'T14':[-1.33, 1.13, 3.677], 'T15':[-0.97, 1.13, 3.657], 'T16':[-0.875, 1.155, 3.657]}


transformsh = {}

for key in rot.keys():
    rotation = rot[key]
    translation = np.array(tran[key])
    mat = np.eye(4)
    #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #R = mesh.get_rotation_matrix_from_xyz((rotation[0], rotation[1], rotation[2]))
    R = rotmat(math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2]))
    mat[:3, :3] = R #make into rotation matrix
    mat[:-1, -1] = translation
    #print("key here is", key)
    transformsh[key] = mat

  

R1h = [-0.70710361, -0.2418444,   0.66446653,  -0.70710996,  0.24184223, -0.66446057, 0. ,  -0.93969337, -0.3420181 ]
R2h = [ 0.25882121, -0.33036391 , 0.90767354, -0.96592524, -0.08852154 , 0.24321258, 0.  , -0.93969337, -0.3420181 ]
R3h = [-0.25882318, -0.33036373,  0.90767305, -0.96592472 , 0.08852221, -0.24321443, 0.  ,  -0.93969337 ,-0.3420181]
R4h = [ 0.70710924, -0.24184248,  0.66446124,-0.70710433, -0.24184416,  0.66446586, 0.,   -0.93969337, -0.3420181 ]


R5h=[[-0.70710736,  0.24184312, -0.66446301],
 [ 0.7071062,   0.24184351, -0.66446409],
 [ 0. ,        -0.93969337, -0.3420181 ]]
R6h=[[-0.96592498, -0.08852188,  0.2432135 ],
 [-0.2588222 ,  0.33036382, -0.9076733 ],
 [ 0. ,        -0.93969337, -0.3420181 ]]
R7h=[[-0.96592635, 0.08852012, -0.24320869],
 [ 0.25881707,  0.33036429, -0.90767459],
 [ 0.,         -0.93969337, -0.3420181 ]]
R8h=[[-0.70710361, -0.2418444,   0.66446653],
 [-0.70710996,  0.24184223, -0.66446057],
 [ 0. ,        -0.93969337, -0.3420181 ]]

R13h = [[ 0.70710924, -0.24184248,  0.66446124],
 [-0.70710433, -0.24184416,  0.66446586],
 [ 0.,         -0.93969337, -0.3420181 ]]
R14h = [[ 0.96592688,  0.08851945, -0.24320684],
 [ 0.2588151,  -0.33036447,  0.90767508],
 [ 0.,         -0.93969337, -0.3420181 ]]
R15h = [[ 0.96592704, -0.08851925,  0.24320628],
 [-0.25881451, -0.33036453,  0.90767523],
 [ 0.,         -0.93969337, -0.3420181 ]]
R16h = [[ 0.7071088,   0.24184263, -0.66446165],
 [ 0.70710476, -0.24184401,  0.66446545],
 [ 0.,         -0.93969337, -0.3420181 ]]


T1h, T2h, T3h, T4h = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
T5h, T6h, T7h, T8h = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
T13h, T14h, T15h, T16h = np.eye(4), np.eye(4), np.eye(4), np.eye(4)

T1h[:3, :3], T2h[:3, :3], T3h[:3, :3], T4h[:3, :3] = np.array(R1h).reshape(3, 3), np.array(R2h).reshape(3, 3), np.array(R3h).reshape(3, 3), np.array(R4h).reshape(3, 3)
T5h[:3, :3], T6h[:3, :3], T7h[:3, :3], T8h[:3, :3] = np.array(R5h).reshape(3, 3), np.array(R6h).reshape(3, 3), np.array(R7h).reshape(3, 3), np.array(R8h).reshape(3, 3)
T13h[:3, :3], T14h[:3, :3], T15h[:3, :3], T16h[:3, :3] = np.array(R13h).reshape(3, 3), np.array(R14h).reshape(3, 3), np.array(R15h).reshape(3, 3), np.array(R16h).reshape(3, 3)

T1h[:-1, 3], T2h[:-1, 3], T3h[:-1, 3], T4h[:-1, 3] = [0.15, 0.355, 3.637], [0.125, 0.255, 3.637], [0.125, -0.255, 3.637], [0.15, -0.355, 3.637]
T5h[:-1, 3], T6h[:-1, 3], T7h[:-1, 3], T8h[:-1, 3] = [-0.875, -1.155, 3.657], [-0.97, -1.13, 3.657], [-1.33, -1.13, 3.677], [-1.425, -1.155, 3.677]
T13h[:-1, 3], T14h[:-1, 3], T15h[:-1, 3], T16h[:-1, 3] = [-1.425, 1.155, 3.677], [-1.33, 1.13, 3.677], [-0.97, 1.13, 3.657], [-0.875, 1.155, 3.657]

transformsh['T01'] = np.linalg.inv(T1h)
transformsh['T02'] = np.linalg.inv(T2h)
transformsh['T03'] = np.linalg.inv(T3h)
transformsh['T04'] = np.linalg.inv(T4h)

transformsh['T05'] = np.linalg.inv(T5h)
transformsh['T06'] = np.linalg.inv(T6h)
transformsh['T07'] = np.linalg.inv(T7h)
transformsh['T08'] = np.linalg.inv(T8h)

transformsh['T13'] = np.linalg.inv(T13h)
transformsh['T14'] = np.linalg.inv(T14h)
transformsh['T15'] = np.linalg.inv(T15h)
transformsh['T16'] = np.linalg.inv(T16h)


def get_transforms(halo = False):
    
    if(halo):
        return transformsh
    else:
        return transformsc
    
    
def plot_fused_cloud(point_cloud, n_left_im):

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud.copy())
    colors = n_left_im
    pc.colors = o3d.utility.Vector3dVector(colors)    

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    #viewer.ground_plane("XZ")
    viewer.add_geometry(pc)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()

    return pc
    
    '''o3d.visualization.draw_geometries([pc],
              zoom=0.3412,
              front=[0.4257, -0.2125, -0.8795],
              lookat=[2.6172, 2.0475, 1.532],
              up=[-0.0694, -0.9768, 0.2024])'''

def get_T(pts, group_info, key, transforms):
    
    '''R = group_info["extrinsics"][key]["r"]
    t = group_info["extrinsics"][key]["t"]
    print(key)
    print(R)
    #print(key, t)
    T = np.eye(4)
    T[:3, :3] = np.array(R).reshape(3, 3)
    T[:-1, 3] = t'''
    if(group_info is not None):
    
        P = np.array(group_info["extrinsics"][key]).reshape(3, 4)
        T = np.eye(4)
        T[:-1] = P
    else:

        T = transforms[key]
            
    
    return T
    
def transform_cloud(cloud, cam_loc, transforms, group_info=None, halo = False, reshape = False):
    pts = o3d.geometry.PointCloud()
    cloud_filt = cloud.copy()
    if(not reshape):
        cloud_filt[cloud_filt[:, -1] > 40] = 0
    pts.points = o3d.utility.Vector3dVector(cloud_filt)

    T = get_T(pts, group_info, cam_loc, transforms) 
    if(cam_loc == "side-left-left" and not halo):
        pitch_rot = (0., 0, 0) #(-0.139626, 0, 0)
        R = pts.get_rotation_matrix_from_xyz(pitch_rot)
        pts_inter = copy.deepcopy(pts).transform(np.linalg.inv(T))
        pts_new = pts_inter.rotate(R, center=(0, 0, 0))
    else:
        pts_new = copy.deepcopy(pts).transform(np.linalg.inv(T))
    
    return np.array(pts_new.points)
    
'''
def transform_all_clouds(all_clouds):
    transformed_clouds = {}
    for i, key in enumerate(all_clouds):       
        new_cloud = transform_cloud(all_clouds[key], key)
        transformed_clouds[key] = new_cloud.copy()
    return transformed_clouds
'''
   
def transform_all_clouds(all_clouds, group_info=None, halo = False, reshape = False, ext_tranforms = None):

    
    if(not halo):
        transforms = transformsc
    else:
        transforms = transformsh
    
    if(ext_transforms is not None):
        transforms = ext_transforms

    transformed_clouds = {}
    for i, key in enumerate(all_clouds):  
        #print("key is", key)

        if(all_clouds[key] is not None):
            new_cloud = transform_cloud(all_clouds[key], key, transforms, group_info, halo, reshape)
            if(not reshape):
                new_cloud[new_cloud[:, -1]<0] = 0
            if(reshape):
                new_cloud = new_cloud.reshape((512, 1024, 3))
            transformed_clouds[key] = new_cloud.copy()
    return transformed_clouds

    
def visualize_fused_cloud(transformed_clouds, all_lefts, i, j):
    #print(transformed_clouds.keys())
    #all_lefts_new = {}
    #for i, key in enumerate(all_lefts.keys()):       
    #    all_lefts_new[key] = np.ones_like(all_lefts[key].copy())*(i+1)*0.1
    #print("length of all lefts", len(list(transformed_clouds.values())))
    fused_cloud = np.concatenate(list(transformed_clouds.values())[i:j], axis = 0)
    fused_colours = np.concatenate(list(all_lefts.values())[i:j], axis = 0)
    print(fused_cloud.shape, fused_colours.shape)
    pc = plot_fused_cloud(fused_cloud, fused_colours)

    return pc


def transform_object(obj, cam_loc, transforms, group_info=None, halo = False):


    T = get_T(obj, group_info, cam_loc, transforms)
    new_obj_tuple = []
    if(cam_loc == "side-left-left" and not halo):
        pitch_rot = (0., 0, 0) #(-0.139626, 0, 0)
        R = obj.get_rotation_matrix_from_xyz(pitch_rot)
        obj_inter = copy.deepcopy(obj).transform(np.linalg.inv(T))
        obj_new = obj_inter.rotate(R, center=(0, 0, 0))
    else:
        obj_new = copy.deepcopy(obj).transform(np.linalg.inv(T))

    bbox = obj_new.get_axis_aligned_bounding_box()
    box_ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    box_ls.paint_uniform_color((0.5, 0, 0))   
    new_obj_tuple = [obj_new, box_ls, bbox] 
    return new_obj_tuple


def transform_all_objects(all_objects, group_info = None, halo = False):
        

    if(not halo):
        transforms = transformsc
    else:
        transforms = transformsh
    
    all_transformed_objects = {}
    all_transformed_boxes = {}
    for _, loc in enumerate(all_objects): 
        stopclass_transformed = {}
        stopclass_boxes = {}
        for _, stopclass in enumerate(all_objects[loc]):

            obj_list = all_objects[loc][stopclass]
            #print(loc, stopclass, obj_list)
            transformed_objects = []
            for object_tuple in obj_list:           
                new_object_tuple = transform_object(object_tuple[0], loc, transforms, group_info, halo)
                transformed_objects.append(new_object_tuple.copy())
                
            if(len(transformed_objects) > 0):
                transformed_boxes = [item[-1] for item in transformed_objects]
            else:
                transformed_boxes = []
            stopclass_transformed[stopclass] = transformed_objects
            stopclass_boxes[stopclass] = transformed_boxes
            
        all_transformed_objects[loc] = stopclass_transformed
        #remove empty stopclass_boxes
        vals = list(stopclass_boxes.values())
        all_empty = True
        for item in vals:
            if(len(item) > 0):
                all_empty = False
        if(all_empty):
            stopclass_boxes = {}
        ##############################   
        #print("loc, stopclass_boxes", loc, stopclass_boxes)
        all_transformed_boxes[loc] = stopclass_boxes
    return all_transformed_objects, all_transformed_boxes


def visualize_fused_objects(transformed_objects, full_clouds, all_lefts, viz = True):
    every_single_object = []
    for _, loc in enumerate(transformed_objects): 
        for _, stopclass in enumerate(transformed_objects[loc]):
            obj_list = transformed_objects[loc][stopclass]

            for object_tuple in obj_list:  
                for object in object_tuple[:-1]:
                    every_single_object.append(object)

    
    
    for _, loc in enumerate(full_clouds):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(full_clouds[loc].copy())
        colors = all_lefts[loc]
        pc.colors = o3d.utility.Vector3dVector(colors)       
        every_single_object.append(pc)

    if(viz):    
        o3d.visualization.draw_geometries(every_single_object,
                  zoom=0.3412,
                  front=[0.4257, -0.2125, -0.8795],
                  lookat=[2.6172, 2.0475, 1.532],
                  up=[-0.0694, -0.9768, 0.2024])

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        #viewer.ground_plane("XZ")
        for item in every_single_object:
            viewer.add_geometry(item)
        
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=4.0)
        viewer.add_geometry(mesh)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        viewer.run()
        viewer.destroy_window()
    return every_single_object