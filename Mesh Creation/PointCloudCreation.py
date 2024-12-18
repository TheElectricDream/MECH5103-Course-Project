# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:15:24 2024

@author: adams
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import Birch
import pyvista as pv
import open3d as o3d
from scipy.spatial.transform import Rotation


def convert_points_to_mesh(points):
    '''
    Parameters
    ----------
    points : NumPy array of size [n_points, 3]
        3D point cloud with [X,Y,Z] coordinates.

    Returns
    -------
    surface : PyVista mesh
        3D triangulated mesh.

    '''
    cloud = pv.PolyData(points)
    volume = cloud.delaunay_2d(alpha=0.05)
    surface = volume.extract_geometry()
    return surface

#%%
if __name__ == '__main__':

    df = pd.read_excel('Data Points_LOCAL_2024_10_16.xlsx',
                       sheet_name='Range Data FINAL')
    
    n_slices = 24 # number of slices
    n_points = 25 # points per slice
    
    distance = np.zeros([n_slices, n_points])
    
    # Read the data into a structured np array
    for i in range(n_slices):
        distance[i,:] = df[f"Slice {i+1}"].to_list()
    
    df_angle = pd.read_excel('Data Points_LOCAL_2024_10_16.xlsx', 
                             sheet_name='Angle Data (Missing 1 Column)')
    
    angle_UD = df_angle["Encoder Angle DXL1 (degrees)"].to_list()
    angle_LR = df_angle["Encoder Angle DXL2 (degrees)"].to_list()
    
    angle_UD[3008] = angle_UD[3009]
    
    # Visualize the data
    x = list(range(len(angle_UD)))
    
    i_1 = 300
    i_2 = 410
    
    # Up-down angles
    plt.figure()
    plt.grid(alpha = 0.45)
    plt.scatter(x[i_1:i_2],
                angle_UD[i_1:i_2],
                50,
                facecolor='k',
                color='k',
                linewidth=1,
                marker='x')
    plt.xlabel("Sample Number")
    plt.ylabel("Up-Down Angle ($^\circ$)")
    plt.tight_layout()
    plt.savefig('Single Elevation Data.svg')
    plt.show()
    
    # Left-right angles
    plt.figure()
    plt.grid(alpha = 0.45)
    plt.scatter(x, angle_LR)
    plt.xlabel("Point Number")
    plt.ylabel("Left-Right Angle (degrees)")
    plt.show()
    
    # Slice 6 (Slice that was missed in first dataset)
    df_angle_s6 = pd.read_excel('Data Points_LOCAL_2024_10_16.xlsx', 
                             sheet_name='Angle Data (Column 6)')
    angle_UD_s6 = df_angle_s6["Encoder Angle DXL1 (degrees)"].to_list()
    angle_LR_s6 = df_angle_s6["Encoder Angle DXL2 (degrees)"].to_list()
    
    x_s6 = list(range(len(angle_UD_s6)))
    
    plt.figure()
    plt.grid(alpha = 0.45)
    plt.scatter(x_s6, angle_UD_s6)
    plt.xlabel("Point Number")
    plt.ylabel("Up-Down Angle (degrees)")
    plt.show()
    
    plt.figure()
    plt.grid(alpha = 0.45)
    plt.scatter(x_s6, angle_LR_s6)
    plt.xlabel("Point Number")
    plt.ylabel("Left-Right Angle (degrees)")
    plt.show()
    
    ''' 
    Just hard code indices for the angles for the L-R measurements. These indices 
    were found by looking at when each step started and ended. This could have 
    been automated, although it only took ~5 minutes to get all the points (vs. 
    the hour or so that it would have taken to automate and check), and it only
    had to be done once.
    '''
    
    angle_LR_mat = np.zeros([n_slices, n_points])
    
    # Slices 1-5
    slice1_start = [1,   140, 290, 450, 560]
    slice1_end   = [120, 250, 410, 540, 650]
    
    
    for i in range(5):
        angle_LR_mat[i,:] = np.average(angle_LR[slice1_start[i]:slice1_end[i]])
    
    # Slice 6
    slice6_start = 100
    slice6_end   = 190
    
    angle_LR_mat[5,:] = np.average(angle_LR_s6[slice6_start:slice6_end])
    
    # Slices 7-end
    slice7_start = [800, 920,  1050, 1175, 1320, 1420, 1575, 1700, 1830, 1975, 2100, \
                    2225, 2350, 2450, 2570, 2770, 2890, 3015]
    slice7_end   = [900, 1020, 1130, 1300, 1400, 1550, 1680, 1810, 1950, 2090, 2200, \
                    2325, 2430, 2550, 2730, 2870, 3000, 3100]
    for i in np.array(range(18)) + 6:
        angle_LR_mat[i,:] = np.average(angle_LR[slice7_start[i-6]:slice7_end[i-6]])
    
    # Look at the U-D measurements 
    
    '''
    Here we automate the up-down angles since there are far more steps.
    '''
    
    angle_UD_mat = np.zeros([n_slices, n_points])
    
    # Slices 1-5
    count = 1
    angle_prev = angle_UD[0]
    angle_UD_plat = []
    going_up = True
    for i in range(5):
        angle_UD_plat.append([])
        while True:
            try:
                angle_temp = angle_UD[count]
            except:
                break
            
            diff = angle_temp - angle_prev
            count += 1
            angle_prev = angle_temp
            if abs(diff) < 1:
                # Point count+1 is on same plateau
                angle_UD_plat[-1].append(angle_prev)
            else:
                # No longer on the same plateau
                angle_UD_plat[-1] = np.average(angle_UD_plat[-1])
                if diff < -0.5 and going_up:
                    # We've switched from going up to going down
                    going_up = False
                    angle_UD_plat.append(angle_UD_plat[-1])
                    break
                elif diff > 0.5 and not going_up:
                    # We've switched from going down to going up
                    going_up = True
                    angle_UD_plat.append(angle_UD_plat[-1])
                    break
                angle_UD_plat.append([])
    
    
    # Check for skipped points and linearly interpolate for them
    for i in range(len(angle_UD_plat)-1):
        diff = angle_UD_plat[i + 1] - angle_UD_plat[i]
        if abs(diff) > 3.5:
            # This is a skipped point, so insert a point
            val = (angle_UD_plat[i + 1] + angle_UD_plat[i])/2
            angle_UD_plat.insert(i+1, val)
    
    # Remove the last item since it's a duplicate
    angle_UD_plat.pop()
    
    angle_UD_mat = np.zeros([n_slices, n_points])
    
    for j in range(5):
        for k in range(n_points):
            angle_UD_mat[j,k] = angle_UD_plat[j*n_points + k]
    
    # Slice 6
    angle_UD_s6_1 = angle_UD_s6[82:-2]
    
    angle_prev = angle_UD_s6_1[0]
    count = 1
    
    angle_UD_plat_s6 = []
    
    angle_UD_plat_s6.append([])
    while True:
        try:
            angle_temp = angle_UD_s6_1[count]
        except:
            angle_UD_plat_s6[-1] = np.average(angle_UD_plat_s6[-1])
            break
        
        diff = angle_temp - angle_prev
        count += 1
        angle_prev = angle_temp
        if abs(diff) < 1:
            # Point count+1 is on same plateau
            angle_UD_plat_s6[-1].append(angle_prev)
        else:
            # No longer on the same plateau
            angle_UD_plat_s6[-1] = np.average(angle_UD_plat_s6[-1])
            angle_UD_plat_s6.append([])
    
    angle_UD_mat[5,:] = angle_UD_plat_s6
    
    # Slices 7-end
    count = 771
    angle_prev = angle_UD[770]
    angle_UD_plat_s7 = []
    going_up = True
    for i in np.array(range(18)) + 6:
        angle_UD_plat_s7.append([])
        while True:
            try:
                angle_temp = angle_UD[count]
            except:
                angle_UD_plat_s7[-1] = np.average(angle_UD_plat_s7[-1])
                break
            
            diff = angle_temp - angle_prev
            count += 1
            angle_prev = angle_temp
            if abs(diff) < 1:
                # Point count+1 is on same plateau
                angle_UD_plat_s7[-1].append(angle_prev)
            else:
                # No longer on the same plateau
                angle_UD_plat_s7[-1] = np.average(angle_UD_plat_s7[-1])
                if diff < -0.5 and going_up:
                    # We've switched from going up to going down
                    going_up = False
                    angle_UD_plat_s7.append(angle_UD_plat_s7[-1])
                    break
                elif diff > 0.5 and not going_up:
                    # We've switched from going down to going up
                    going_up = True
                    angle_UD_plat_s7.append(angle_UD_plat_s7[-1])
                    break
                angle_UD_plat_s7.append([])
    
    
    # Check for skipped points and linearly interpolate for them
    for i in range(len(angle_UD_plat_s7)-1):
        diff = angle_UD_plat_s7[i + 1] - angle_UD_plat_s7[i]
        if abs(diff) > 3.5:
            # This is a skipped point, so insert a point
            val = (angle_UD_plat_s7[i + 1] + angle_UD_plat_s7[i])/2
            angle_UD_plat_s7.insert(i+1, val)
    
    
    for j in np.array(range(18)) + 6:
        for k in range(n_points):
            angle_UD_mat[j,k] = angle_UD_plat_s7[(j-6)*n_points + k]
    
    
    # [distance, up-down angle, left_right angle]
    coord_dist_ang = np.zeros([n_slices, n_points, 3])
    
    for i in range(n_slices):
        for j in range(n_points):
            coord_dist_ang[i,j,:] = np.array([distance[i,j] - 0.0396875, 
                                     -angle_UD_mat[i,j] +270-1,
                                     angle_LR_mat[i,j] + 182.99-6])
    ''' 
    Now that we have all the individual measurements organized in spherical,
    coordinates, we can convert into cartesian coordinates, with the form:
    X is fore-aft towards/away from the objects (normal to back wall)
    Y is left-right along the table (LR)
    Z is the height (UD)
    '''
    coord_xyz = np.zeros([n_slices, n_points, 3])
    
    for i in range(n_slices):
        for j in range(n_points):
            x = coord_dist_ang[i,j,0] *\
                np.sin(np.deg2rad(coord_dist_ang[i,j,1])) *\
                np.cos(np.deg2rad(coord_dist_ang[i,j,2])) + 0.1539875
            y = coord_dist_ang[i,j,0] *\
                np.sin(np.deg2rad(coord_dist_ang[i,j,1])) *\
                np.sin(np.deg2rad(coord_dist_ang[i,j,2])) + 0.6683375
            z = coord_dist_ang[i,j,0] *\
                np.cos(np.deg2rad(coord_dist_ang[i,j,1])) + 0.1095375
            
            coord_xyz[i,j,:] = [x,y,z]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(0,0,0, s=250, marker='+', label='Corner of Table')
    ax.scatter(0.1539875, 0.6683375, 0.1095375, 
               s=250, marker='x', label="Lidar Source")
    plt.legend()
    for i in range(n_slices):
        ax.scatter(coord_xyz[i,:,0], coord_xyz[i,:,1], coord_xyz[i,:,2])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.show()
    
    plt.figure()
    plt.grid(alpha=0.45)
    plt.scatter(coord_xyz[:,:, 0], coord_xyz[:,:,1])
    plt.axis('equal')
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.show()
    
    plt.figure()
    plt.grid(alpha=0.45)
    plt.scatter(coord_xyz[:,:, 0], coord_xyz[:,:,2])
    plt.axis('equal')
    plt.xlabel("X Position (m)")
    plt.ylabel("Z Position (m)")
    plt.show()
    
    plt.figure()
    plt.grid(alpha=0.45)
    plt.scatter(coord_xyz[:,:, 1], coord_xyz[:,:,2])
    plt.axis('equal')
    plt.xlabel("Y Position (m)")
    plt.ylabel("Z Position (m)")
    plt.show()
    
    # Now overlay the points from the stereo images
    
    with open(r'point_clouds.pkl', 'rb') as f:
        stereo_data = pickle.load(f)
    
    #%%
    
    fig = plt.figure(figsize=[7,7])
    fig.tight_layout()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(0,0,0, label='Corner of Table (World Origin)', s=100)
    ax.scatter(0.1539875, 0.6683375, 0.1095375, label="Lidar Source (Centre of Rotation)", s=100)
    plt.legend()
    for i in range(n_slices):
        ax.scatter(coord_xyz[i,:,0], coord_xyz[i,:,1], coord_xyz[i,:,2],
                   alpha = 0.5, s = 15)
    ax.set_box_aspect([1,1,1]) 
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.view_init(elev=30, azim=-135, roll=0)
    # plt.savefig('point_cloud_1.svg')
    plt.show()
    
    #%% Automate point cloud to mesh conversion
    
    # Prepare dataset for open3d by converting into an open3d point cloud
    xyz = coord_xyz.reshape(
        [coord_xyz.shape[0]*coord_xyz.shape[1], 3]) # [n_points, 3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # RANSAC for 3d plane segmentation
    
    pt_to_plane_dist = 0.005
    max_plane_idx = 2 # Floor and wall are the most significant planes
    
    segment_models = {}
    segments = {}
    segments_np = {}
    
    colours = [ [1,0,0], [0,1,0], [0,0,1],
               [1,0.5,0], [1,0,1], [0,1,1],
               [0.6, 0.6, 0.6]]
    
    surface_labels = ["Wall",
                      "Floor",
                      "Wood Block", 
                      "Tall Box", 
                      "Tissue Box", 
                      "Back Wall",
                      "Outlier"]
    
    #%%
    
    remaining = pcd # points remaining; not included in another plane
    
    for i in range(max_plane_idx):
        # Segment out the largest plane in the remaining point cloud
        segment_models[i], inliers = remaining.segment_plane(
            distance_threshold=pt_to_plane_dist,
            ransac_n=3,
            num_iterations=10000)
        # Save the segment and colour them
        segments[i] = remaining.select_by_index(inliers)
        segments[i].paint_uniform_color(colours[i])
        # Remove the points were selected
        remaining = remaining.select_by_index(inliers, invert=True)
        segments_np[surface_labels[i]] = (np.asarray(segments[i].points))
    
    # Paint remaining points grey so we can see them
    remaining.paint_uniform_color([0.6, 0.6, 0.6])
    
    # Visualize the segments
    o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [remaining]) # 
    
    #%%
    o3d.visualization.draw_geometries([remaining])
    
    #%% Save the point clouds
    
    # o3d.io.write_point_cloud("point_cloud_wall.ply", segments[0])
    # o3d.io.write_point_cloud("point_cloud_floor.ply", segments[1])
    # o3d.io.write_point_cloud("point_cloud_remaining.ply", remaining)
    
    #%% Read the point clouds
    
    segments = {}
    
    segments[0] = o3d.io.read_point_cloud("point_cloud_wall.ply")
    segments[1] = o3d.io.read_point_cloud("point_cloud_floor.ply")
    remaining = o3d.io.read_point_cloud("point_cloud_remaining.ply")
    
    # o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [remaining])
    
    wall_np = np.asarray(segments[0].points)
    floor_np = np.asarray(segments[1].points)
    remaining_np_1 = np.asarray(remaining.points)
    
    fig = plt.figure(figsize=[7,7])
    fig.tight_layout()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(wall_np[:,0], wall_np[:,1], wall_np[:,2],
               alpha = 0.5, s = 15, facecolor='r',
               label="Wall")
    ax.scatter(floor_np[:,0], floor_np[:,1], floor_np[:,2],
               alpha = 0.5, s = 15, facecolor='g',
               label="Floor")
    ax.scatter(remaining_np_1[:,0], remaining_np_1[:,1], remaining_np_1[:,2],
               alpha = 0.5, s = 15, facecolor=[0.3,0.3,0.3],
               label="Uncategorized")
    plt.legend()
    ax.set_box_aspect([1,1,1]) 
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.view_init(elev=45, azim=210, roll=0)
    # plt.savefig('point_cloud_segmented.svg')
    plt.show()
    
    #%%
    # Now cluster the remaining 4 surfaces +  1 outlier
    
    n_clusters = 5
    
    remaining_np = np.asarray(remaining.points)
    
    # Cluster only along x-y axes since the objects are aligned with the z-axis
    remaining_np_xy = remaining_np[:,0:2]
    
    cluster_model = Birch(threshold=0.001, n_clusters=n_clusters)
    cluster_model.fit(remaining_np_xy)
    yhat = cluster_model.predict(remaining_np_xy)
    clusters = np.unique(yhat)
    
    wall = np.asarray(segments[0].points)
    floor = np.asarray(segments[1].points)
    
    wood_block = remaining_np[np.where(yhat == clusters[0])]
    back_wall = remaining_np[np.where(yhat == clusters[1])]
    tall_box = remaining_np[np.where(yhat == clusters[2])]
    tissue_box = remaining_np[np.where(yhat == clusters[3])]
    outlier = remaining_np[np.where(yhat == clusters[4])]
    
    surfaces_array = [wall,
                floor,
                wood_block, 
                tall_box, 
                tissue_box,
                back_wall,
                outlier]
    
    fig = plt.figure(figsize=[7,7])
    fig.tight_layout()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(wall_np[:,0], wall_np[:,1], wall_np[:,2],
               alpha = 0.5, s = 15, facecolor='r',
               label="Wall")
    ax.scatter(floor_np[:,0], floor_np[:,1], floor_np[:,2],
               alpha = 0.5, s = 15, facecolor='g',
               label="Floor")
    ax.scatter(wood_block[:,0], wood_block[:,1], wood_block[:,2],
               alpha = 0.5, s = 15, facecolor='b',
               label="Wood Block")
    ax.scatter(tall_box[:,0], tall_box[:,1], tall_box[:,2],
               alpha = 0.5, s = 15, facecolor='violet',
               label="Tall Box")
    ax.scatter(tissue_box[:,0], tissue_box[:,1], tissue_box[:,2],
               alpha = 0.5, s = 15, facecolor='cyan',
               label="Tissue Box")
    ax.scatter(back_wall[:,0], back_wall[:,1], back_wall[:,2],
               alpha = 0.5, s = 15, facecolor='orange',
               label="Back Wall")
    ax.scatter(outlier[:,0], outlier[:,1], outlier[:,2],
               alpha = 0.5, s = 15, facecolor=[0.3,0.3,0.3],
               label="Outlier")
    plt.legend()
    ax.set_box_aspect([1,1,1]) 
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.view_init(elev=45, azim=210, roll=0)
    # plt.savefig('point_cloud_segmented_clustered.svg')
    plt.show()
    
    
    for i in range(len(surface_labels)):
        segments_np[surface_labels[i]] = surfaces_array[i]
        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(surfaces_array[i])
        segments[i] = pcd_1
        segments[i].paint_uniform_color(colours[i])
    
    o3d.visualization.draw_geometries([segments[i] for i in range(len(segments))] ) # + [remaining]
    
    # Let's make a mesh now using PyVista
    
    back_wall_mesh = convert_points_to_mesh(back_wall)
    wall_mesh = convert_points_to_mesh(segments_np['Wall'])
    floor_mesh = convert_points_to_mesh(segments_np['Floor'])
    wood_block_mesh = convert_points_to_mesh(wood_block)
    tall_box_mesh = convert_points_to_mesh(tall_box)
    tissue_box_mesh = convert_points_to_mesh(tissue_box)
    
    # 3D Visualization
    
    plotter = pv.Plotter()
    plotter.add_mesh(back_wall_mesh, show_edges=True)
    plotter.add_mesh(wall_mesh, show_edges=True)
    plotter.add_mesh(floor_mesh, show_edges=True)
    plotter.add_mesh(wood_block_mesh, show_edges=True)
    plotter.add_mesh(tall_box_mesh, show_edges=True)
    plotter.add_mesh(tissue_box_mesh, show_edges=True)
    
    plotter.show()
    
    
    #%% Get validation data's camera rotation and position
    
    with open('point_clouds_rev7.pkl', 'rb') as f:
        stereo_data_1 = pickle.load(f)
    
    r1 = Rotation.from_matrix(stereo_data_1['R1'])
    r2 = Rotation.from_matrix(stereo_data_1['R2'])
    
    # Convert to Euler angles to be inputted into Blender
    angles1 = r1.as_euler('xyz', degrees=True)
    angles2 = r2.as_euler('xyz', degrees=True)
    
    # Get positions for Blender
    t1 = stereo_data_1['t1']
    t2 = stereo_data_1['t2']
    
    #%%
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1]) 
    # ax.scatter(0,0,0, label='Corner of Table', s=100)
    # ax.scatter(0.1539875, 0.6683375, 0.1095375, label="Lidar Source", s=100)
    ax.scatter(np.asarray(segments[0].points)[:,0], np.asarray(segments[0].points)[:,1], np.asarray(segments[0].points)[:,2], label='Wall')
    ax.scatter(back_wall[:,0], back_wall[:,1], back_wall[:,2], label='Far Wall')
    ax.scatter(np.asarray(segments[1].points)[:,0], np.asarray(segments[1].points)[:,1], np.asarray(segments[1].points)[:,2], label='Floor')
    ax.scatter(wood_block[:,0], wood_block[:,1], wood_block[:,2], label='Wood Block')
    ax.scatter(tall_box[:,0], tall_box[:,1], tall_box[:,2], label='Tall Box')
    ax.scatter(tissue_box[:,0], tissue_box[:,1], tissue_box[:,2], label='Tissue Box')
    ax.scatter(outlier[:,0], outlier[:,1], outlier[:,2], label='Outlier')
    plt.legend()
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.show()