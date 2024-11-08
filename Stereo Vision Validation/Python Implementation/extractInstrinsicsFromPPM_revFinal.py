import numpy as np
import pickle
import plotly.graph_objects as go
import cv2
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
from scipy.linalg import rq

# Disable scientific notation and set precision
np.set_printoptions(suppress=True,  # Suppress scientific notation
                   precision=4,      # Number of decimal places
                   floatmode='fixed')  # Fixed number of decimal places


def triangulate_points(M1, M2, point1s, point2s):
    """
    Triangulate 3D points from corresponding 2D points in stereo images.
    Parameters:
    M1 (numpy.ndarray): The 3x4 projection matrix for the first camera.
    M2 (numpy.ndarray): The 3x4 projection matrix for the second camera.
    point1s (numpy.ndarray): An Nx2 array of 2D points from the first image.
    point2s (numpy.ndarray): An Nx2 array of 2D points from the second image.
    Returns:
    numpy.ndarray: An Nx3 array of triangulated 3D points.
    """

    world_points = np.array([])
    for i in range(point1s.shape[0]):

        point1 = point1s[i]
        point2 = point2s[i]

        # Construct matrix A from the projection equations
        A = np.array([
            (point1[0] * M1[2, :] - M1[0, :]),
            (point1[1] * M1[2, :] - M1[1, :]),
            (point2[0] * M2[2, :] - M2[0, :]),
            (point2[1] * M2[2, :] - M2[1, :])
        ])
        
        # Solve for the 3D point using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]  # Solution is the last row of Vt
        X = X / X[3]  # Convert to non-homogeneous coordinates (normalize by the last coordinate)
        if world_points.size == 0:
            world_points = X[:3].reshape(1, 3)
        else:
            world_points = np.vstack((world_points, X[:3].reshape(1, 3)))

        
    return world_points


def return_intrinsic_properties_6_points(P1, P2, P3, P4, P5, P6, U, V):
    """
    Calculate the intrinsic properties of a camera given a set of 3D points and their corresponding 2D projections.
    Parameters:
    P1, P2, P3, P4, P5, P6 : array-like
        Arrays representing the 3D coordinates of six points in the world space.
    U, V : array-like
        Arrays representing the 2D coordinates of the projections of the six points in the image plane.
    Returns:
    M : ndarray
        The 3x4 transformation matrix.
    K : ndarray
        The 3x3 intrinsic matrix.
    R : ndarray
        The 3x3 rotation matrix.
    t : ndarray
        The translation vector.
    """

    # Calculate the parameters of the A matrix for camera 
    A = np.array([[P1[0], P1[1], P1[2], 1, 0, 0, 0, 0, -U[0]*P1[0], -U[0]*P1[1], -U[0]*P1[2], -U[0]],
                  [0, 0, 0, 0, P1[0], P1[1], P1[2], 1, -V[0]*P1[0], -V[0]*P1[1], -V[0]*P1[2], -V[0]],
                  [P2[0], P2[1], P2[2], 1, 0, 0, 0, 0, -U[1]*P2[0], -U[1]*P2[1], -U[1]*P2[2], -U[1]],
                  [0, 0, 0, 0, P2[0], P2[1], P2[2], 1, -V[1]*P2[0], -V[1]*P2[1], -V[1]*P2[2], -V[1]],
                  [P3[0], P3[1], P3[2], 1, 0, 0, 0, 0, -U[2]*P3[0], -U[2]*P3[1], -U[2]*P3[2], -U[2]],
                  [0, 0, 0, 0, P3[0], P3[1], P3[2], 1, -V[2]*P3[0], -V[2]*P3[1], -V[2]*P3[2], -V[2]],
                  [P4[0], P4[1], P4[2], 1, 0, 0, 0, 0, -U[3]*P4[0], -U[3]*P4[1], -U[3]*P4[2], -U[3]],
                  [0, 0, 0, 0, P4[0], P4[1], P4[2], 1, -V[3]*P4[0], -V[3]*P4[1], -V[3]*P4[2], -V[3]],
                  [P5[0], P5[1], P5[2], 1, 0, 0, 0, 0, -U[4]*P5[0], -U[4]*P5[1], -U[4]*P5[2], -U[4]],
                  [0, 0, 0, 0, P5[0], P5[1], P5[2], 1, -V[4]*P5[0], -V[4]*P5[1], -V[4]*P5[2], -V[4]],
                  [P6[0], P6[1], P6[2], 1, 0, 0, 0, 0, -U[5]*P6[0], -U[5]*P6[1], -U[5]*P6[2], -U[5]],
                  [0, 0, 0, 0, P6[0], P6[1], P6[2], 1, -V[5]*P6[0], -V[5]*P6[1], -V[5]*P6[2], -V[5]]])

    # Calculate the SVD
    SVD_U, SVD_S, SVD_Vh = np.linalg.svd(A)

    # Need to transpose Vh to get V (Numpy specific)
    SVD_V = np.transpose(SVD_Vh)

    # Assemble the transformation matrix
    M = np.array([[SVD_V[0,-1], SVD_V[1,-1], SVD_V[2,-1], SVD_V[3,-1]],
                    [SVD_V[4,-1], SVD_V[5,-1], SVD_V[6,-1], SVD_V[7,-1]],
                    [SVD_V[8,-1], SVD_V[9,-1], SVD_V[10,-1], SVD_V[11,-1]]])

    # Decompose M into K, R, and t
    K, R = rq(M[:3,:3])
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    M_r = np.linalg.inv(K) @ M
    t = -np.linalg.inv(M_r[:3,:3]) @ M_r[:, -1]
    #t, residuals, rank, s = np.linalg.lstsq(M[:, :3], -M[:, 3], rcond=None)

    return M, K, R, t


def return_intrinsic_properties_n_points(P, U, V):
    """
    Calculate the intrinsic properties of a camera given a set of 3D points and their corresponding 2D projections.
    Parameters:
    P : array-like
        An Nx3 array representing the 3D coordinates of points in the world space.
    U, V : array-like
        Arrays representing the 2D coordinates of the projections of the points in the image plane.
    Returns:
    M : ndarray
        The 3x4 transformation matrix.
    K : ndarray
        The 3x3 intrinsic matrix.
    R : ndarray
        The 3x3 rotation matrix.
    t : ndarray
        The translation vector.
    """
    num_points = len(P)

    # Construct the A matrix for the system of equations
    A = []
    for i in range(num_points):
        X, Y, Z = P[i]
        u, v = U[i], V[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    A = np.array(A)

    # Solve for M using SVD
    _, _, Vh = np.linalg.svd(A)

  # Need to transpose Vh to get V (Numpy specific)
    SVD_V = np.transpose(Vh)

    # Assemble the transformation matrix
    M = np.array([[SVD_V[0,-1], SVD_V[1,-1], SVD_V[2,-1], SVD_V[3,-1]],
                    [SVD_V[4,-1], SVD_V[5,-1], SVD_V[6,-1], SVD_V[7,-1]],
                    [SVD_V[8,-1], SVD_V[9,-1], SVD_V[10,-1], SVD_V[11,-1]]])


    K, R = rq(M[:3,:3])
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    M_r = np.linalg.inv(K) @ M
    t = -np.linalg.inv(M_r[:3,:3]) @ M_r[:, -1]

    return M, K, R, t


# Main function
if __name__ == "__main__":

    # Hard code the points extracted using the selectPointsForPPM.py script

    # These ponts are for rev1, which was the tissue box
    # U1 = np.array([1705.84, 1693.90, 1643.19, 1469.28, 1504.77, 1528.02])
    # V1 = np.array([1366.94, 1091.92, 997.93, 1013.47, 1104.05, 1387.08])
    # U2 = np.array([1906.03, 1905.87, 1966.18, 1799.79, 1729.24, 1734.55])
    # V2 = np.array([1351.90, 1077.88, 989.02, 975.90, 1061.55, 1337.05])

    # # These points are for rev2, which was the wooden block
    # U11 = np.array([1976.92,2187.64,2338.49,2348.13,2132.12,1985.96])
    # V11 = np.array([1324.48,1424.20,1361.11,1169.20,1078.72,1127.85])
    # U21 = np.array([2200.42,2308.06,2522.20,2538.69,2413.71,2219.78])
    # V21 = np.array([1366.44,1503.29,1476.75,1271.74,1145.84,1167.13])

    # U12 = np.array([1976.80,2187.08,2340.91,2349.40,2131.14,1984.80])
    # V12 = np.array([1322.41,1423.80,1361.86,1168.57,1077.67,1127.62])
    # U22 = np.array([2201.04,2307.45,2522.91,2540.38,2413.33,2217.98])
    # V22 = np.array([1364.96,1502.60,1479.83,1275.49,1144.73,1168.55])

    # U13 = np.array([1977.84, 2187.13, 2339.50, 2348.01, 2131.56, 1985.91])
    # V13 = np.array([1322.87, 1421.91, 1362.31, 1170.05, 1077.74, 1127.93])
    # U23 = np.array([2199.22, 2307.44, 2523.89, 2539.15, 2411.50, 2218.19])
    # V23 = np.array([1362.58, 1503.18, 1478.66, 1272.40, 1145.22, 1167.88])

    # # Take the average of the points
    # U1 = (U11 + U12 + U13) / 3
    # V1 = (V11 + V12 + V13) / 3
    # U2 = (U21 + U22 + U23) / 3
    # V2 = (V21 + V22 + V23) / 3

    # These points are for revFinal, which is a mix of points measured in blender
    U1 = np.array([994.04, 1130.81, 1323.45, 1527.57, 1643.08, 1984.40, 1706.82, 1691.76])
    V1 = np.array([639.72, 1722.25, 1589.26, 1387.17, 997.37, 1129.09, 1365.43,1092.20])
    U2 = np.array([1030.41, 1149.69, 1398.14, 1734.31, 1966.41, 2217.97,1905.45,1904.88])
    V2 = np.array([540.94, 1548.72, 1473.72, 1337.16, 989.52, 1169.19,1350.05,1078.66])

    # Hard code the location of points in 3D space

    # These points are for rev1, which was the tissue box
    # P1 = np.array([0, 0, 0])
    # P2 = np.array([0, 0, 0.120])
    # P3 = np.array([0.120, 0, 0.120])
    # P4 = np.array([0.120, 0.073, 0.120])
    # P5 = np.array([0, 0.073, 0.120])
    # P6 = np.array([0, 0.073, 0])

    # These points are for rev2, which was the wooden block
    # P1 = np.array([0, 0, 0])
    # P2 = np.array([0.128, 0, 0])
    # P3 = np.array([0.128, 0.087, 0])
    # P4 = np.array([0.128, 0.087, 0.087])
    # P5 = np.array([0, 0.087, 0.087])
    # P6 = np.array([0, 0, 0.087])

    # These points are for revFinal, which is a mix of points measured in blender
    P1 = np.array([0.42409, 0.88457, 0.37098])
    P2 = np.array([0.42409, 0.88457, 0])
    P3 = np.array([0.49632, 0.82061, 0])
    P4 = np.array([0.63478, 0.73245, 0])
    P5 = np.array([0.75478, 0.65945, 0.120])
    P6 = np.array([0.65714, 0.53881, 0.087])
    P7 = np.array([0.63478, 0.65945, 0])
    P8 = np.array([0.63478, 0.65945, 0.120])

    # From the checkerboard calibration we got the following intrinsic parameters
    K_Checkerboard = np.array([[3190.76177614524,	0,	                2014.68354646223],
                               [0,	                3180.94277382243,	1499.72323418057],
                               [0,	                0,	                1]])

    # From knowing the parameters of the camera we got the following intrinsic parameters
    K_Actual = np.array([[3263,	    0,	    2016],
                         [0,	    3263,	1512],
                         [0,	    0,	    1]])

    # Stack the real point locations
    P = np.vstack((P1, P2, P3, P4, P5, P6, P7, P8))

    # Run the calibration
    M1, K1, R1, t1 = return_intrinsic_properties_n_points(P, U1, V1)
    M2, K2, R2, t2 = return_intrinsic_properties_n_points(P, U2, V2)

    # Define the points to triangulate 
    UV1_WOOD = np.array([[2188.37, 1423.95], [2339.91, 1362.11], [2348.88, 1170.91], [2132.19, 1077.44],
                            [1985.37, 1128.43], [1976.87, 1323.40], [2186.48, 1423.48], [2203.00, 1221.43],
                            [1984.43, 1128.43], [2203.00, 1221.90], [2276.65, 1284.22], [2348.88, 1170.44]])

    UV2_WOOD = np.array([[2307.72, 1503.12], [2523.73, 1478.27], [2540.91, 1274.45], [2413.92, 1145.20],
                            [2219.14, 1167.35], [2199.71, 1364.83], [2306.36, 1503.12], [2331.22, 1294.34],
                            [2219.14, 1167.35], [2331.67, 1294.79], [2429.28, 1372.07], [2539.55, 1273.10]])

    UV1_TISSUE = np.array([[1527.02, 1387.59], [1706.76, 1364.80], [1692.08, 1092.92], [1642.96, 996.72],
                            [1468.29, 1012.92], [1504.24, 1109.12], [1526.51, 1387.59], [1692.58, 1092.92],
                            [1503.73, 1109.12], [1706.76, 1364.80], [1692.58, 1092.41], [1468.29, 1012.42]])
    UV2_TISSUE = np.array([[1733.87, 1336.65], [1905.76, 1350.86], [1903.93, 1079.96], [1966.73, 988.74],
                            [1798.05, 975.45], [1726.08, 1063.92], [1733.87, 1336.65], [1905.31, 1079.50],
                            [1726.54, 1063.92], [1905.76, 1351.32], [1904.85, 1079.50], [1798.50, 975.45]])

    UV1_BOX = np.array([[1131.01, 1719.99], [1324.83, 1585.92], [1212.62, 568.71], [658.83, 485.64],
                         [416.92, 543.93], [988.19, 634.29], [1131.01, 1718.54], [1214.07, 565.79],
                         [991.10, 634.29], [1324.83, 1585.92]])
    UV2_BOX = np.array([[1150.12, 1546.38], [1399.88, 1474.83], [1316.63, 505.68], [1031.74, 389.90],
                            [775.47, 417.22], [1030.44, 538.20], [1147.51, 1546.38], [1319.23, 503.08],
                            [1031.74, 539.50], [1401.19, 1474.83]])

    # These points are special cases for additional validation
    # UV1_LIDAR = np.array([[1968.05, 1559.88]])
    # UV2_LIDAR = np.array([[1458.19, 1525.89]])
    # UV1_WALL = np.array([[2605.04, 694.93], [2367.25, 1126.04]])
    # UV2_WALL = np.array([[3245.15, 833.20], [2745.68, 1246.69]])

    # Triangulate 3D points from 2D correspondences
    P_WOOD    = triangulate_points(M1, M2, UV1_WOOD, UV2_WOOD)
    P_TISSUE  = triangulate_points(M1, M2, UV1_TISSUE, UV2_TISSUE)
    P_BOX     = triangulate_points(M1, M2, UV1_BOX, UV2_BOX)
    P_CAL     = np.array([P1, P2, P3, P4, P5, P6, P7, P8])

    # P_WALL  = triangulate_points(M1, M2, UV1_WALL, UV2_WALL)
    # P_LIDAR = triangulate_points(M1, M2, UV1_LIDAR, UV2_LIDAR)

    # The world origin is located at:
    WORLD_ORIGIN  = np.array([0, 0, 0])

    # The frame being used in the final code IS the world frame, so no translations or rotations are required
    ROTATION = 0
    R_PPM_TO_WORLD = np.array([[np.cos(np.radians(ROTATION)), -np.sin(np.radians(ROTATION)), 0],
                               [np.sin(np.radians(ROTATION)),  np.cos(np.radians(ROTATION)), 0],
                               [0,                             0,                            1]])
    

    # Rotate the points to align with the world coordinate system
    P_WOOD   = P_WOOD @ R_PPM_TO_WORLD.T  
    P_TISSUE = P_TISSUE @ R_PPM_TO_WORLD.T
    P_BOX    = P_BOX @ R_PPM_TO_WORLD.T
    P_CAL    = P_CAL  @ R_PPM_TO_WORLD.T

    # P_WALL = P_WALL @ R_PPM_TO_WORLD.T
    # P_LIDAR = P_LIDAR @ R_PPM_TO_WORLD.T

    # Translate the points to the world origin
    P_WOOD   = P_WOOD   + WORLD_ORIGIN
    P_TISSUE = P_TISSUE + WORLD_ORIGIN
    P_BOX    = P_BOX    + WORLD_ORIGIN
    P_CAL    = P_CAL    + WORLD_ORIGIN

    # Rotate the cameras to align with the world coordinate system
    R1 = R1 @ R_PPM_TO_WORLD.T
    R2 = R2 @ R_PPM_TO_WORLD.T 
    
    # Transpose to get the correct orientation
    R1 = R1.T
    R2 = R2.T

    # Translate the cameras to the world origin
    t1 = t1.flatten() + WORLD_ORIGIN
    t2 = t2.flatten() + WORLD_ORIGIN

    # Rotate the cameras to align with the world coordinate system
    t1 = t1 @ R_PPM_TO_WORLD.T
    t2 = t2 @ R_PPM_TO_WORLD.T

    # Save the point clouds to a pickle file
    with open('point_clouds.pkl', 'wb') as f:
        pickle.dump({'P_CAL': P_CAL, 'P_WOOD': P_WOOD, 'P_TISSUE': P_TISSUE, 'P_BOX': P_BOX, 't1': t1, 't2': t2, 'R1':R1, 'R2': R2}, f)

    # Initialize Plotly figure
    fig = go.Figure()
    scale = 0.1

    x, y, z = zip(*P_WOOD)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5, color='green'), name="Points (WOOD)"))

    x, y, z = zip(*P_TISSUE)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5, color='blue'), name="Points (TISSUE)"))

    x, y, z = zip(*P_BOX)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5, color='orange'), name="Points (BOX)"))

    x, y, z = zip(*P_CAL)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='red'), line=dict(color='red'), name="Points (CAL)"))

    # x, y, z = zip(*P_WALL)
    # fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='blue'), name="Points (WALLS)"))

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='blue', symbol='x'), name="ORIGIN"))

    # Plot cameras' positions
    fig.add_trace(go.Scatter3d(x=[t1[0]], y=[t1[1]], z=[t1[2]], mode='markers', marker=dict(size=8, color='blue'), name="Camera 1"))
    fig.add_trace(go.Scatter3d(x=[t2[0]], y=[t2[1]], z=[t2[2]], mode='markers', marker=dict(size=8, color='green'), name="Camera 2"))

    # Plot Camera 1 axes
    for i, color in enumerate(['red', 'green', 'blue']):
        fig.add_trace(go.Scatter3d(
            x=[t1[0], t1[0] + scale * R1[0, i]], 
            y=[t1[1], t1[1] + scale * R1[1, i]], 
            z=[t1[2], t1[2] + scale * R1[2, i]], 
            mode='lines', line=dict(color=color), name=f"Camera 1 {['X', 'Y', 'Z'][i]}-axis"
        ))

    # Plot Camera 2 axes
    for i, color in enumerate(['red', 'green', 'blue']):
        fig.add_trace(go.Scatter3d(
            x=[t2[0], t2[0] + scale * R2[0, i]], 
            y=[t2[1], t2[1] + scale * R2[1, i]], 
            z=[t2[2], t2[2] + scale * R2[2, i]], 
            mode='lines', line=dict(color=color, dash='dash'), name=f"Camera 2 {['X', 'Y', 'Z'][i]}-axis"
        ))

    # Configure axes and layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="3D Points and Camera Orientation"
    )

    # Show the plot
    fig.show()