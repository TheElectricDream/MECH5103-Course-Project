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

def normalize_2D_points(points):

    # Calculate centroid
    centroid = np.mean(points, axis=0)

    # Calculate average distance to centroid
    avg_distance = np.mean(np.linalg.norm(points - centroid, axis=1))

    # Scale points to have average distance sqrt(2)
    scale = np.sqrt(2) / avg_distance
    T = np.array([[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]])

    # Apply transformation to points
    points_normalized = np.hstack((points, np.ones((points.shape[0], 1)))) @ T.T

    return points_normalized, T


def normalize_3D_points(points):

    # Calculate centroid
    centroid = np.mean(points, axis=0)

    # Calculate average distance to centroid
    avg_distance = np.mean(np.linalg.norm(points - centroid, axis=1))

    # Scale points to have average distance sqrt(3)
    scale = np.sqrt(3) / avg_distance
    T = np.array([[scale, 0, 0, -scale * centroid[0]], [0, scale, 0, -scale * centroid[1]], [0, 0, scale, -scale * centroid[2]], [0, 0, 0, 1]])

    # Apply transformation to points
    points_normalized = np.hstack((points, np.ones((points.shape[0], 1)))) @ T.T

    return points_normalized, T


def reprojection_error(params, points_3D, points_2D_1, points_2D_2, M1, M2):
    """
    Calculate the reprojection error for 3D points projected onto 2D image planes of two cameras.

    Args:
        params (np.ndarray): Flattened array of 3D points.
        points_3D (np.ndarray): Original 3D points.
        points_2D_1 (np.ndarray): Corresponding 2D points in the first camera.
        points_2D_2 (np.ndarray): Corresponding 2D points in the second camera.
        M1 (np.ndarray): Projection matrix for the first camera.
        M2 (np.ndarray): Projection matrix for the second camera.

    Returns:
        np.ndarray: Flattened array of reprojection errors for both cameras.
    """
    # Reshape parameters to get 3D points
    n_points = points_3D.shape[0]
    points_3D = params.reshape((n_points, 3))

    # Initialize error list
    error = []

    # Loop through each 3D point
    for i in range(n_points):
        # Convert 3D point to homogeneous coordinates
        point_3D_homogeneous = np.append(points_3D[i], 1)
        
        # Reproject onto Camera 1 using M1
        projected_1 = M1 @ point_3D_homogeneous
        projected_1 /= projected_1[2]  # Normalize to get pixel coordinates
        error.append(projected_1[:2] - points_2D_1[i])  # Calculate error for Camera 1

        # Reproject onto Camera 2 using M2
        projected_2 = M2 @ point_3D_homogeneous
        projected_2 /= projected_2[2]  # Normalize to get pixel coordinates
        error.append(projected_2[:2] - points_2D_2[i])  # Calculate error for Camera 2

    # Return flattened array of errors for least_squares optimization
    return np.array(error).ravel()


def bundle_adjustment_direct(params, n_points, points_2D_1, points_2D_2):
    """
    Perform bundle adjustment by directly optimizing the reprojection error.
    Parameters:
    -----------
    params : array-like
        A 1D array containing the flattened 3D points and the projection matrices.
        The first `n_points * 3` elements are the 3D points, followed by 12 elements
        for the first projection matrix (M1) and 12 elements for the second projection
        matrix (M2).
    n_points : int
        The number of 3D points.
    points_2D_1 : array-like
        A 2D array of shape (n_points, 2) containing the 2D coordinates of the points
        in the first image.
    points_2D_2 : array-like
        A 2D array of shape (n_points, 2) containing the 2D coordinates of the points
        in the second image.
    Returns:
    --------
    error : ndarray
        A 1D array containing the reprojection error for each point in both images,
        flattened for use with optimization routines such as `scipy.optimize.least_squares`.
    """
    # Unpack the parameters
    points_3D = params[:n_points * 3].reshape((n_points, 3))
    M1 = params[n_points * 3 : n_points * 3 + 12].reshape((3, 4))
    M2 = params[n_points * 3 + 12 : n_points * 3 + 24].reshape((3, 4))

    # Calculate reprojection error
    error = []
    for i in range(n_points):
        point_3D_homogeneous = np.append(points_3D[i], 1)
        
        # Project onto Camera 1 using M1
        projected_1 = M1 @ point_3D_homogeneous
        projected_1 /= projected_1[2]  # Normalize to get pixel coordinates
        error.append(projected_1[:2] - points_2D_1[i])

        # Project onto Camera 2 using M2
        projected_2 = M2 @ point_3D_homogeneous
        projected_2 /= projected_2[2]  # Normalize to get pixel coordinates
        error.append(projected_2[:2] - points_2D_2[i])

    return np.array(error).ravel()  # Flatten for least_squares


def bundle_adjustment_scalar(params, n_points, points_2D_1, points_2D_2):
    """
    Computes the total reprojection error for a bundle adjustment problem.
    Parameters:
    -----------
    params : array-like
        A 1D array containing the flattened 3D points and the projection matrices M1 and M2.
        The first n_points * 3 elements correspond to the 3D points.
        The next 12 elements correspond to the flattened 3x4 projection matrix M1.
        The next 12 elements correspond to the flattened 3x4 projection matrix M2.
    n_points : int
        The number of 3D points.
    points_2D_1 : array-like
        A 2D array of shape (n_points, 2) containing the 2D coordinates of the points in the first image.
    points_2D_2 : array-like
        A 2D array of shape (n_points, 2) containing the 2D coordinates of the points in the second image.
    Returns:
    --------
    float
        The total reprojection error, which is the sum of squared differences between the observed
        2D points and the projected 3D points for both cameras.
    """
    # Unpack parameters
    points_3D = params[:n_points * 3].reshape((n_points, 3))
    M1 = params[n_points * 3 : n_points * 3 + 12].reshape((3, 4))
    M2 = params[n_points * 3 + 12 : n_points * 3 + 24].reshape((3, 4))

    # Calculate total reprojection error (sum of squared errors)
    total_error = 0.0
    for i in range(n_points):
        point_3D_homogeneous = np.append(points_3D[i], 1)
        
        # Project onto Camera 1 using M1
        projected_1 = M1 @ point_3D_homogeneous
        projected_1 /= projected_1[2]  # Normalize to get pixel coordinates
        error_1 = np.sum((projected_1[:2] - points_2D_1[i]) ** 2)

        # Project onto Camera 2 using M2
        projected_2 = M2 @ point_3D_homogeneous
        projected_2 /= projected_2[2]  # Normalize to get pixel coordinates
        error_2 = np.sum((projected_2[:2] - points_2D_2[i]) ** 2)

        # Accumulate squared error for both cameras
        total_error += error_1 + error_2

    return total_error  # Return scalar sum of squared errors


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


def return_intrinsic_properties(P1, P2, P3, P4, P5, P6, U, V):
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

    # # Normalize the points
    # P3D, T3D = normalize_3D_points(np.array([P1, P2, P3, P4, P5, P6]))
    # P2D, T2D = normalize_2D_points(np.array([U, V]).T)

    # # Extract the points
    # P1 = P3D[0]
    # P2 = P3D[1]
    # P3 = P3D[2]
    # P4 = P3D[3]
    # P5 = P3D[4]
    # P6 = P3D[5]
    # U = P2D[:, 0]
    # V = P2D[:, 1]

    # Normalize the points
    # U = U / np.abs(U).max()
    # V = V / np.abs(V).max()

    # PX = np.array([P1[0], P2[0], P3[0], P4[0], P5[0], P6[0]])
    # PY = np.array([P1[1], P2[1], P3[1], P4[1], P5[1], P6[1]])
    # PZ = np.array([P1[2], P2[2], P3[2], P4[2], P5[2], P6[2]])

    # # Normalize P
    # SCALE = np.abs(np.array([PX, PY, PZ])).max()
    # PX = PX / SCALE
    # PY = PY / SCALE
    # PZ = PZ / SCALE

    # # Extract P1, P2, P3, P4, P5, P6
    # P1 = np.array([PX[0], PY[0], PZ[0]])
    # P2 = np.array([PX[1], PY[1], PZ[1]])
    # P3 = np.array([PX[2], PY[2], PZ[2]])
    # P4 = np.array([PX[3], PY[3], PZ[3]])
    # P5 = np.array([PX[4], PY[4], PZ[4]])
    # P6 = np.array([PX[5], PY[5], PZ[5]])


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


    world_points = np.array([P1, P2, P3, P4, P5, P6])
    projections = np.column_stack((U, V))
        
    geometric_error(M.ravel(), world_points, projections)

    [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = \
    fmin_bfgs(geometric_error, 
              M.ravel(), 
              args=(world_points, projections), 
              maxiter=2000, 
              full_output=True, 
              retall=False)
    
    # M = xopt.reshape(3, 4)   
    K, R = rq(M[:3,:3])
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    M_r = np.linalg.inv(K) @ M
    t = -np.linalg.inv(M_r[:3,:3]) @ M_r[:, -1]
    #t, residuals, rank, s = np.linalg.lstsq(M[:, :3], -M[:, 3], rcond=None)

    return M, K, R, t


def geometric_error(m, world_points, projections):
    """
    Calculate the geometric error between the projected points and the actual points.
    Args:
        m (numpy.ndarray): A 1D array of 12 elements representing the projection matrix.
        world_points (numpy.ndarray): A 2D array of shape (n_points, 3) containing the 3D coordinates of the world points.
        projections (numpy.ndarray): A 2D array of shape (n_points, 2) containing the 2D coordinates of the projected points.
    Returns:
        float: The total geometric error.
    Notes:
        The function computes the reprojection error for each point and sums them up.
        The reprojection error is the Euclidean distance between the actual projected point and the estimated projected point.
    """
    # https://towardsdatascience.com/camera-calibration-with-example-in-python-5147e945cdeb
    
    error = 0
    n_points = len(world_points)
    for i in range(n_points):
        X, Y, Z = world_points[i, :]
        u, v = projections[i,:]
        u_ = m[0] * X + m[1] * Y + m[2] * Z + m[3]
        v_ = m[4] * X + m[5] * Y + m[6] * Z + m[7]
        d = m[8] * X + m[9] * Y + m[10] * Z + m[11]
        u_ = u_/d
        v_ = v_/d
        error += np.sqrt((u - u_)**2 + (v - v_)**2)

    return error

# Main function
if __name__ == "__main__":

    # Hard code the points extracted using the selectPointsForPPM.py script

    # These ponts are for rev1, which was the tissue box
    # U1 = np.array([1705.84, 1693.90, 1643.19, 1469.28, 1504.77, 1528.02])
    # V1 = np.array([1366.94, 1091.92, 997.93, 1013.47, 1104.05, 1387.08])
    # U2 = np.array([1906.03, 1905.87, 1966.18, 1799.79, 1729.24, 1734.55])
    # V2 = np.array([1351.90, 1077.88, 989.02, 975.90, 1061.55, 1337.05])

    # These points are for rev2, which was the wooden block
    U11 = np.array([1976.92,2187.64,2338.49,2348.13,2132.12,1985.96])
    V11 = np.array([1324.48,1424.20,1361.11,1169.20,1078.72,1127.85])
    U21 = np.array([2200.42,2308.06,2522.20,2538.69,2413.71,2219.78])
    V21 = np.array([1366.44,1503.29,1476.75,1271.74,1145.84,1167.13])

    U12 = np.array([1976.80,2187.08,2340.91,2349.40,2131.14,1984.80])
    V12 = np.array([1322.41,1423.80,1361.86,1168.57,1077.67,1127.62])
    U22 = np.array([2201.04,2307.45,2522.91,2540.38,2413.33,2217.98])
    V22 = np.array([1364.96,1502.60,1479.83,1275.49,1144.73,1168.55])

    U13 = np.array([1977.84, 2187.13, 2339.50, 2348.01, 2131.56, 1985.91])
    V13 = np.array([1322.87, 1421.91, 1362.31, 1170.05, 1077.74, 1127.93])
    U23 = np.array([2199.22, 2307.44, 2523.89, 2539.15, 2411.50, 2218.19])
    V23 = np.array([1362.58, 1503.18, 1478.66, 1272.40, 1145.22, 1167.88])

    # Take the average of the points
    U1 = (U11 + U12 + U13) / 3
    V1 = (V11 + V12 + V13) / 3
    U2 = (U21 + U22 + U23) / 3
    V2 = (V21 + V22 + V23) / 3

    # Hard code the location of points in 3D space

    # These points are for rev1, which was the tissue box
    # P1 = np.array([0, 0, 0])
    # P2 = np.array([0, 0, 0.120])
    # P3 = np.array([0.120, 0, 0.120])
    # P4 = np.array([0.120, 0.073, 0.120])
    # P5 = np.array([0, 0.073, 0.120])
    # P6 = np.array([0, 0.073, 0])

    # These points are for rev2, which was the wooden block
    P1 = np.array([0, 0, 0])
    P2 = np.array([0.128, 0, 0])
    P3 = np.array([0.128, 0.087, 0])
    P4 = np.array([0.128, 0.087, 0.087])
    P5 = np.array([0, 0.087, 0.087])
    P6 = np.array([0, 0, 0.087])

    # From the checkerboard calibration we got the following intrinsic parameters
    K_Checkerboard = np.array([[3190.76177614524,	0,	                2014.68354646223],
                               [0,	                3180.94277382243,	1499.72323418057],
                               [0,	                0,	                1]])

    # From knowing the parameters of the camera we got the following intrinsic parameters
    K_Actual = np.array([[3263,	    0,	    2016],
                         [0,	    3263,	1512],
                         [0,	    0,	    1]])

    # Calculate intrinsic properties for both cameras
    M1, K1, R1, t1 = return_intrinsic_properties(P1, P2, P3, P4, P5, P6, U1, V1)
    M2, K2, R2, t2 = return_intrinsic_properties(P1, P2, P3, P4, P5, P6, U2, V2)

    # Decompose the projection matrices into intrinsic and extrinsic properties
    cameraParams1 = cv2.decomposeProjectionMatrix(M1)
    cameraParams2 = cv2.decomposeProjectionMatrix(M2)

    K1 = cameraParams1[0]
    R1 = cameraParams1[1]
    t1 = cameraParams1[2]
    t1 = t1[:3] / t1[3]
    t1 = t1.reshape(1,3).flatten()

    K2 = cameraParams2[0]
    R2 = cameraParams2[1]
    t2 = cameraParams2[2]
    t2 = t2[:3] / t2[3]
    t2 = t2.reshape(1,3).flatten()

    Rt1 = np.linalg.inv(K_Checkerboard) @ M1
    Rt2 = np.linalg.inv(K_Checkerboard) @ M2

    R1_new = Rt1[:, :3]
    R2_new = Rt2[:, :3]

    # Calculate the SVD
    SVD_U1, SVD_S1, SVD_V1h = np.linalg.svd(R1_new)
    SVD_U2, SVD_S2, SVD_V2h = np.linalg.svd(R2_new)

    SVD_S1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    R1 = SVD_U1 @ SVD_S1 @ SVD_V1h
    R2 = SVD_U2 @ SVD_S1 @ SVD_V2h

    # Rotate 180 around X axis to get the Z matrix pointing in the right direction
    ROT = 180
    R_x = np.array([[1,                       0,                        0], 
                    [0, np.cos(np.radians(ROT)), -np.sin(np.radians(ROT))], 
                    [0, np.sin(np.radians(ROT)),  np.cos(np.radians(ROT))]])
    R2 = R_x @ R2

    # K1s = K1/K1[2,2]
    # K2s = K2/K2[2,2]

    # T = np.diag(np.sign(np.diag(K1s)))
    # K1 = K1 @ T
    # R1 = T @ R1

    # T = np.diag(np.sign(np.diag(K2s)))
    # K2 = K2 @ T
    # R2 = T @ R2

    # Define arrays containing new points that are in the image and were not used for calibration
    # UV1_NEW = np.array([[1968.05, 1559.88], [2202.76, 1220.26], [2187.66, 1424.09], 
    #                     [1980.20, 1324.09], [1125.65, 1722.11], [1323.53, 1589.71], [1986.17, 1128.66], 
    #                     [1215.53, 573.22], [2130.39, 1078.89], [2348.14, 1170.25], 
    #                     [2275.35, 1282.49], [2339.22, 1362.33], [1705.84, 1366.94], [1693.90, 1091.92],
    #                     [1643.19, 997.93], [1469.28, 1013.47], [1504.77, 1104.05], [1528.02, 1387.08]]) 
    # UV2_NEW = np.array([[1458.19, 1525.89], [2331.92, 1294.59], [2307.73, 1504.32], 
    #                     [2202.91, 1369.19], [1144.02, 1544.47], [1398.63, 1473.58], [2218.13, 1168.49], 
    #                     [1318.51, 505.89], [2413.08, 1145.99], [2539.26, 1272.87],
    #                     [2431.68, 1374.62], [2522.58, 1479.28], [1906.03, 1351.90], [1905.87, 1077.88],
    #                     [1966.18, 989.02], [1799.79, 975.90], [1729.24, 1061.55], [1734.55, 1337.05]])


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


    # UV1_TISSUE = np.array([[1705.84, 1366.94], [1693.90, 1091.92], [1643.19, 997.93], [1469.28, 1013.47],
    #                        [1504.77, 1104.05], [1528.02, 1387.08]])
    # UV2_TISSUE = np.array([[1906.03, 1351.90], [1905.87, 1077.88], [1966.18, 989.02], [1799.79, 975.90],
    #                        [1729.24, 1061.55], [1734.55, 1337.05]])

    UV1_BOX = np.array([[1131.01, 1719.99], [1324.83, 1585.92], [1212.62, 568.71], [658.83, 485.64],
                         [416.92, 543.93], [988.19, 634.29], [1131.01, 1718.54], [1214.07, 565.79],
                         [991.10, 634.29], [1324.83, 1585.92], [1131.01, 1717.08], [1214.07, 568.71]])
    UV2_BOX = np.array([[1150.12, 1546.38], [1399.88, 1474.83], [1316.63, 505.68], [1031.74, 389.90],
                            [775.47, 417.22], [1030.44, 538.20], [1147.51, 1546.38], [1319.23, 503.08],
                            [1031.74, 539.50], [1401.19, 1474.83], [1319.23, 503.08], [1150.12, 1543.78]])


    # These points were specifically special
    # UV1_LIDAR = np.array([[1968.05, 1559.88]])
    # UV2_LIDAR = np.array([[1458.19, 1525.89]])
    # UV1_WALL = np.array([[2605.04, 694.93], [2367.25, 1126.04]])
    # UV2_WALL = np.array([[3245.15, 833.20], [2745.68, 1246.69]])

    # Triangulate 3D points from 2D correspondences
    P_WOOD    = triangulate_points(M1, M2, UV1_WOOD, UV2_WOOD)
    P_TISSUE  = triangulate_points(M1, M2, UV1_TISSUE, UV2_TISSUE)
    P_BOX     = triangulate_points(M1, M2, UV1_BOX, UV2_BOX)
    P_CAL     = np.array([P1, P2, P3, P4, P5, P6])

    # P_WOOD = np.empty([])
    # # Create a loop and triangulate all the points using cv2
    # for i in range(UV1_WOOD.shape[0]):
    #     points4D = cv2.triangulatePoints(M1, M2, np.array([UV1_WOOD[i]]).T, np.array([UV2_WOOD[i]]).T)
    #     P_WOOD_NORM = points4D[:3]/points4D[3]
    #     if i == 0:
    #         P_WOOD = P_WOOD_NORM.T
    #     else:
    #         P_WOOD = np.vstack((P_WOOD, P_WOOD_NORM.T))

    # P_TISSUE = np.empty([])
    # # Create a loop and triangulate all the points using cv2
    # for i in range(UV1_TISSUE.shape[0]):
    #     points4D = cv2.triangulatePoints(M1, M2, np.array([UV1_TISSUE[i]]).T, np.array([UV2_TISSUE[i]]).T)
    #     P_TISSUE_NORM = points4D[:3]/points4D[3]
    #     if i == 0:
    #         P_TISSUE = P_TISSUE_NORM.T
    #     else:
    #         P_TISSUE = np.vstack((P_TISSUE, P_TISSUE_NORM.T))

    # P_BOX = np.empty([])
    # # Create a loop and triangulate all the points using cv2
    # for i in range(UV1_BOX.shape[0]):
    #     points4D = cv2.triangulatePoints(M1, M2, np.array([UV1_BOX[i]]).T, np.array([UV2_BOX[i]]).T)
    #     P_BOX_NORM = points4D[:3]/points4D[3]
    #     if i == 0:
    #         P_BOX = P_BOX_NORM.T
    #     else:
    #         P_BOX = np.vstack((P_BOX, P_BOX_NORM.T))

    # P_WALL  = triangulate_points(M1, M2, UV1_WALL, UV2_WALL)
    # P_LIDAR = triangulate_points(M1, M2, UV1_LIDAR, UV2_LIDAR)

    # The world origin is located at:
    WORLD_ORIGIN_BLOCK  = np.array([0.65714, 0.53881, 0.008137])  # This is relative to [0,0,0] which is coincident with P1
    WORLD_ORIGIN_TISSUE = np.array([0.634, 0.675, 0])

    # The frame that is attached to P1 is 25 degrees + 180 degrees rotated counterclockwise about the z-axis
    ROTATION = 25+180
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
    P_WOOD   = P_WOOD   + WORLD_ORIGIN_BLOCK
    P_TISSUE = P_TISSUE + WORLD_ORIGIN_BLOCK
    P_BOX    = P_BOX    + WORLD_ORIGIN_BLOCK
    P_CAL    = P_CAL    + WORLD_ORIGIN_BLOCK

    # Rotate the cameras to align with the world coordinate system
    R1 = R1 @ R_PPM_TO_WORLD.T
    R2 = R2 @ R_PPM_TO_WORLD.T 
    
    # Transpose to get the correct orientation
    R1 = R1.T
    R2 = R2.T

    # Translate the cameras to the world origin
    t1 = t1.flatten() - WORLD_ORIGIN_BLOCK
    t2 = t2.flatten() - WORLD_ORIGIN_BLOCK

    # Rotate the cameras to align with the world coordinate system
    t1 = t1 @ R_PPM_TO_WORLD.T
    t2 = t2 @ R_PPM_TO_WORLD.T

    # Scale the distance by the first element of K to get the correct units
    t1 = t1*K1[0,0]
    t2 = t2*K2[0,0] 

    # Save the point clouds to a pickle file
    # with open('point_clouds.pkl', 'wb') as f:
    #     pickle.dump({'P_CAL': P_CAL, 'P_NEW': P_NEW, 'P_WALL': P_WALL, 't1': t1, 't2': t2, 'R1':R1, 'R2': R2}, f)

    with open('point_clouds.pkl', 'wb') as f:
        pickle.dump({'P_CAL': P_CAL, 'P_WOOD': P_WOOD, 'P_TISSUE': P_TISSUE, 'P_BOX': P_BOX, 't1': t1, 't2': t2, 'R1':R1, 'R2': R2}, f)

    # Initialize Plotly figure
    fig = go.Figure()
    scale = 0.1

    # Plot the 3D points
    x, y, z = zip(*P_CAL)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5, color='red'), line=dict(color='red'), name="Points (CAL)"))

    x, y, z = zip(*P_WOOD)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5, color='green'), name="Points (WOOD)"))

    x, y, z = zip(*P_TISSUE)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5, color='blue'), name="Points (TISSUE)"))

    x, y, z = zip(*P_BOX)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5, color='orange'), name="Points (BOX)"))


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
