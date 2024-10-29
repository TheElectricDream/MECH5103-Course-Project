# Load in the saved calibration data and load the fundamental matrix

import numpy as np
import cv2
import pickle
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import least_squares

# Disable scientific notation and set precision
np.set_printoptions(suppress=True,  # Suppress scientific notation
                   precision=4,      # Number of decimal places
                   floatmode='fixed')  # Fixed number of decimal places

def triangulate_points(P1, P2, x1, x2):

    A = np.zeros((4,4))
    A[0,:] = x1[0]*P1[2,:] - P1[0,:]
    A[1,:] = x1[1]*P1[2,:] - P1[1,:]
    A[2,:] = x2[0]*P2[2,:] - P2[0,:]
    A[3,:] = x2[1]*P2[2,:] - P2[1,:]

    # Calculate the SVD
    U, S, VH = np.linalg.svd(A)
    V = VH.T

    # Extract the 3D point
    X = V[:,-1]

    # Normalize the point
    X = X[:3]/X[-1]

    return X


def project(K, R_vec, t, point_3d):
    # This function does the reverse calculation - i.e. going from 3D point back to 2D image plane
    # Convert 3D point to homogeneous coordinates
    point_3d_homogeneous = np.hstack((point_3d, 1))  # [X, Y, Z, 1]

    R, _ = cv2.Rodrigues(R_vec)  # Convert Rodrigues to rotation matrix

    # Apply extrinsic transformation (rotation and translation)
    extrinsic = np.hstack((R, t.reshape(3, 1)))  # Shape (3,4)
    point_cam = extrinsic @ point_3d_homogeneous

    # Project onto image plane using intrinsic matrix K
    point_img = K @ point_cam

    # Convert from homogeneous to 2D by dividing by z-coordinate
    u = point_img[0] / point_img[2]
    v = point_img[1] / point_img[2]

    return np.array([u, v])


def residuals(params, K1, K2, points_2d_cam1, points_2d_cam2):
    # This function calculates the error between the observed 2D points and the reprojected 2D points
    # This is then minimized using least squares optimization to refine the camera parameters and 3D points

    num_points = len(points_2d_cam1)
    cam1_params = params[:6]  # 3 for rotation, 3 for translation
    cam2_params = params[6:12]  # 3 for rotation, 3 for translation
    points_3d = params[12:].reshape((num_points, 3))  # Reshape into (n_points,3)

    # Decompose camera parameters
    R1, t1 = cam1_params[:3], cam1_params[3:]
    R2, t2 = cam2_params[:3], cam2_params[3:]

    residuals = []
    for i in range(num_points):
        # Project the 3D points to 2D for both cameras
        reprojected_2d_cam1 = project(K1, R1, t1, points_3d[i])
        reprojected_2d_cam2 = project(K2, R2, t2, points_3d[i])

        # Calculate residuals (observed - reprojected)
        residuals.append(points_2d_cam1[i] - reprojected_2d_cam1)
        residuals.append(points_2d_cam2[i] - reprojected_2d_cam2)

    return np.array(residuals).ravel()


if __name__ == "__main__":

    # Load the camera calibration data
    with open('calibration_data.pkl', 'rb') as f:
        calibration_data = pickle.load(f)

    # Load the fundamental matrix
    with open('fundamental_matrix.pkl', 'rb') as f:
        F = pickle.load(f)

    # Hard code the points extract using the selectPointsForFundamentalMatrix.py script
    U1 = np.array([1025.97, 1147.92, 1471.66, 1726.62, 1967.05, 1920.81, 2308.22, 2414.17]) #, 1590.47, 1632.56])
    V1 = np.array([539.96, 1543.05, 1473.21, 1062.46, 988.92, 1386.99, 1503.24, 1145.77]) #, 1595.54, 2033.59])
    U2 = np.array([1022.68, 1132.70, 1429.26, 1770.82, 2067.17, 1927.17, 2177.51, 2419.28]) #, 985.09, 1168.71])
    V2 = np.array([626.06, 1539.33, 1526.08, 1174.05, 1133.43, 1496.75, 1666.24, 1330.60]) #, 1579.98, 2021.77])
    
    # Approximate scale factor for the camera calibration
    scale = 2.2594

    # Extract the instrinsic matrices
    K1 = calibration_data['K1']
    K2 = calibration_data['K2']

    # Calculate the essential matrix
    E = K2.T @ F @ K1

    # Enforce the constraint that the essential matrix has a rank of 2
    U_ESVD, S_ESVD, VT_ESVD = np.linalg.svd(E)
    S_ESVD[2] = 0
    E = U_ESVD @ np.diag(S_ESVD) @ VT_ESVD

    # Extract the rotation and translation components using the SVD
    U_ESVD, S_ESVD, VT_ESVD = np.linalg.svd(E)

    # Calculate the possible rotation and translation matrices
    translations = {}
    translations['t'] = U_ESVD[:,2]

    # Calculate the possible rotation matrices
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    # Get the rotation matrix for a THETA degree rotation about the y-axis
    THETA = -0*np.pi/180
    R_y = np.array([[np.cos(THETA), 0, np.sin(THETA)],
                    [0, 1, 0],
                    [-np.sin(THETA), 0, np.cos(THETA)]])
    
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(THETA), -np.sin(THETA)],
                    [0, np.sin(THETA), np.cos(THETA)]])
    
    R_z = np.array([[np.cos(THETA), -np.sin(THETA), 0],
                    [np.sin(THETA), np.cos(THETA), 0],
                    [0, 0, 1]])

    rotations = {}
    rotations['R1'] = U_ESVD @ W @ VT_ESVD 
    rotations['R2'] = U_ESVD @ W.T @ VT_ESVD 
    rotations['R3'] = -U_ESVD @ W @ VT_ESVD 
    rotations['R4'] = -U_ESVD @ W.T @ VT_ESVD 

    # Only use the rotation matrices that have a determinant of 1
    #rotations_true = {}
    #for key in rotations.keys():
    #    if np.linalg.det(rotations[key]) > 0:
    #        rotations_true[key] = rotations[key]

    # Pick a rotation and translation pair
    R = rotations['R4'] @ R_y
    t = translations['t']

    # Calculate the possible projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = np.hstack((R, np.reshape(t,(3,1))))

    # Calculate the 3D points
    points = {}

    for i in range(np.size(U1,0)):

        # Extract the points
        x1 = np.array([U1[i], V1[i]])
        x2 = np.array([U2[i], V2[i]])

        # Add homogeneous coordinate
        x1 = np.hstack((x1, 1))
        x2 = np.hstack((x2, 1))

        # Normalize the points
        x1 = np.reshape(np.linalg.pinv(K1) @ x1, (3,1))
        x2 = np.reshape(np.linalg.pinv(K2) @ x2, (3,1))

        # Calculate the 3D points
        points[f'P{i+1}'] = triangulate_points(P1, P2, x1, x2)

    # Extract the points
    X = [points[f'P{i+1}'][0] for i in range(np.size(U1, 0))]
    Y = [points[f'P{i+1}'][1] for i in range(np.size(U1, 0))]
    Z = [points[f'P{i+1}'][2] for i in range(np.size(U1, 0))]

    # Initial parameters for optimization (use R, t, and triangulated points)
    # Convert to a rodrigez vector
    R_vec, _ = cv2.Rodrigues(R)

    initial_params = np.hstack([np.zeros((1,3)).ravel(), np.zeros((1,3)).ravel(), R_vec.ravel(), t, np.vstack(list(points.values())).ravel()])

    # Run bundle adjustment per https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    result = least_squares(
        residuals,
        initial_params,
        verbose=2,
        args=(K1, K2, np.vstack([U1, V1]).T, np.vstack([U2, V2]).T),
    )

    R1_refined = cv2.Rodrigues(result.x[:3])[0]
    t1_refined = result.x[3:6]
    R2_refined = cv2.Rodrigues(result.x[6:9])[0]
    t2_refined = result.x[9:12]
    points_refined = result.x[12:].reshape(-1, 3)

    # Scale the point distances using a real value we measured
    X = points_refined[:, 0]/scale
    Y = points_refined[:, 1]/scale
    Z = points_refined[:, 2]/scale

    # Define the original axis vectors
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # Rotate each axis vector
    rotated_x1_axis = R1_refined @ x_axis
    rotated_y1_axis = R1_refined @ y_axis
    rotated_z1_axis = R1_refined @ z_axis

    rotated_x2_axis = R2_refined @ x_axis
    rotated_y2_axis = R2_refined @ y_axis
    rotated_z2_axis = R2_refined @ z_axis

    # Create the scatter plot for the points
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z, mode='markers', marker=dict(size=5, color='black'), name='Points'
    ))

    # Camera positions
    fig.add_trace(go.Scatter3d(
        x=[t1_refined[0]], y=[t1_refined[0]], z=[t1_refined[0]], mode='markers', marker=dict(size=6, color='red'), name='Camera 1'
    ))
    fig.add_trace(go.Scatter3d(
        x=[t2_refined[0]], y=[t2_refined[1]], z=[t2_refined[2]], mode='markers', marker=dict(size=6, color='blue'), name='Camera 2'
    ))

    # Set labels and equal aspect ratio
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube'
    ))

    # Show the plot
    fig.show()


    






