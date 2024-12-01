import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_point_clouds(file_path):
    """
    Load point cloud data from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        dict: Loaded point cloud data.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def set_axes_equal(ax):
    """
    Make axes of a 3D plot have equal scale.

    Args:
        ax: A Matplotlib 3D axis object.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_middle = sum(x_limits) / 2
    y_middle = sum(y_limits) / 2
    z_middle = sum(z_limits) / 2

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def fit_plane(points):
    """
    Fit a plane to a set of points using PCA.

    Args:
        points (ndarray): Array of points (Nx3).

    Returns:
        tuple: Normal vector and centroid of the plane.
    """
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Perform PCA to find the normal vector
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    normal = pca.components_[-1]  # Normal vector corresponds to the smallest variance component
    return normal, centroid


def point_to_plane_distance(point, normal, centroid):
    """
    Compute the shortest distance from a point to a plane.

    Args:
        point (ndarray): A single 3D point.
        normal (ndarray): Normal vector of the plane.
        centroid (ndarray): A point on the plane.

    Returns:
        float: Distance from the point to the plane.
    """
    return np.abs(np.dot(point - centroid, normal)) / np.linalg.norm(normal)



if __name__ == "__main__":
    # Load static point clouds
    static_data = load_point_clouds('point_clouds_10pts.pkl')
    P_CAL = static_data['P_CAL']
    P_WOOD = static_data['P_WOOD']
    P_TISSUE = static_data['P_TISSUE']
    P_BOX = static_data['P_BOX']

    # Combine all point clouds into one array
    P = np.vstack((P_CAL, P_WOOD, P_TISSUE, P_BOX))

    # Load dynamic point clouds
    dynamic_data = load_point_clouds('point_cloud_surfaces_lidar.pkl')
    P_WOOD_M = dynamic_data['Wood Block']
    P_BOX_M = dynamic_data['Tall Box']
    P_TISSUE_M = dynamic_data['Tissue Box']

    # Select specific faces of each object
    P_BOX_FACE = P_BOX[[0, 1, 2, 5], :]
    P_TISSUE_FACE = P_TISSUE[[5, 2, 1, 0], :]
    P_WOOD_FACE_1 = P_WOOD[[0, 1, 2, 7], :]
    P_WOOD_FACE_2 = P_WOOD[[0, 4, 5, 7], :]

    # Split dynamic wood block point cloud into two parts
    P_WOOD_M_2 = P_WOOD_M[10:, :]
    P_WOOD_M_1 = P_WOOD_M[:10, :]

    # Create 3D scatter plot
    plt.rcParams.update({'font.size': 14})  # Set font size globally

    # Initialize figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P_TISSUE_FACE[:, 0], P_TISSUE_FACE[:, 1], P_TISSUE_FACE[:, 2], c='r', marker='o')
    ax.scatter(P_BOX_FACE[:, 0], P_BOX_FACE[:, 1], P_BOX_FACE[:, 2], c='r', marker='o')
    ax.scatter(P_WOOD_FACE_1[:, 0], P_WOOD_FACE_1[:, 1], P_WOOD_FACE_1[:, 2], c='r', marker='o')
    ax.scatter(P_WOOD_FACE_2[:, 0], P_WOOD_FACE_2[:, 1], P_WOOD_FACE_2[:, 2], c='r', marker='o')
    ax.scatter(P_TISSUE_M[:, 0], P_TISSUE_M[:, 1], P_TISSUE_M[:, 2], c='orange', marker='o', label='Tissue Box (LiDAR)')
    ax.scatter(P_BOX_M[:, 0], P_BOX_M[:, 1], P_BOX_M[:, 2], c='g', marker='o', label='Tall Box (LiDAR)')
    ax.scatter(P_WOOD_M_1[:, 0], P_WOOD_M_1[:, 1], P_WOOD_M_1[:, 2], c='b', marker='o', label='Wood Block 1 (LiDAR)')
    ax.scatter(P_WOOD_M_2[:, 0], P_WOOD_M_2[:, 1], P_WOOD_M_2[:, 2], c='magenta', marker='o', label='Wood Block 2 (LiDAR)')
    
    ax.plot_trisurf(P_BOX_FACE[:,0], P_BOX_FACE[:,1], P_BOX_FACE[:,2],
                            linewidth = 0.2, 
                            antialiased = True,
                            edgecolor = 'grey',
                            color = 'green',
                            alpha = 0.2,
                            label='Tall Box (Face #1)') 
    
    P_TISSUE_NEW = np.vstack((P_TISSUE_FACE[2,:], P_TISSUE_FACE[0,:], P_TISSUE_FACE[3,:]))

    ax.plot_trisurf(P_TISSUE_NEW[:,0], P_TISSUE_NEW[:,1], P_TISSUE_NEW[:,2],
                            linewidth = 0.2, 
                            antialiased = True,
                            edgecolor = 'grey',
                            color = 'orange',
                            alpha = 0.2,
                            label='Tissue (Face #2)') 

    P_TISSUE_NEW = np.vstack((P_TISSUE_FACE[2,:], P_TISSUE_FACE[1,:], P_TISSUE_FACE[0,:]))

    ax.plot_trisurf(P_TISSUE_NEW[:,0], P_TISSUE_NEW[:,1], P_TISSUE_NEW[:,2],
                            linewidth = 0.2, 
                            antialiased = True,
                            edgecolor = 'grey',
                            color = 'orange',
                            alpha = 0.2) 
    
    ax.plot_trisurf(P_WOOD_FACE_1[:,0], P_WOOD_FACE_1[:,1], P_WOOD_FACE_1[:,2],
                            linewidth = 0.2, 
                            antialiased = True,
                            edgecolor = 'grey',
                            color = 'blue',
                            alpha = 0.2,
                            label='Wood Block (Face #3)') 
    
    ax.plot_trisurf(P_WOOD_FACE_2[:,0], P_WOOD_FACE_2[:,1], P_WOOD_FACE_2[:,2],
                            linewidth = 0.2, 
                            antialiased = True,
                            edgecolor = 'grey',
                            color = 'magenta',
                            alpha = 0.2,
                            label='Wood Block (Face #4)')    
    
    # Define the vertices of the surface
    verts = [[P_BOX_FACE[0], P_BOX_FACE[1], P_BOX_FACE[2], P_BOX_FACE[3]]]

    ax.plot_trisurf(P_BOX_FACE[:,0], P_BOX_FACE[:,1], P_BOX_FACE[:,2],
                            linewidth = 0.2, 
                            antialiased = True,
                            edgecolor = 'grey',
                            color = 'green',
                            alpha = 0.2) 
    

    # Equalize axes
    set_axes_equal(ax)

    # Fit planes and compute errors
    for face, measured in [
        (P_BOX_FACE, P_BOX_M),
        (P_TISSUE_FACE, P_TISSUE_M),
        (P_WOOD_FACE_1, P_WOOD_M_1),
        (P_WOOD_FACE_2, P_WOOD_M_2)
    ]:
        normal, centroid = fit_plane(face)
        errors = [point_to_plane_distance(p, normal, centroid) for p in measured]
        print(f'Mean error for face: {np.mean(errors) * 1000:.2f} mm')

    ax.view_init(elev=90, azim=190)

    # Configure axes and layout
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    #ax.set_title("3D Points and Camera Orientation")
    ax.legend(loc='upper left', bbox_to_anchor=(0.7, 0.9), borderaxespad=0)

    plt.savefig('LiDAR_versus_Stereo_TopView_10pts.svg', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()



