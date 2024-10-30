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
    
    M = xopt.reshape(3, 4)
    K, R = rq(M[:3,:3])
    t, residuals, rank, s = np.linalg.lstsq(M[:, :3], -M[:, 3], rcond=None)

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
    U1 = np.array([1705.84, 1693.90, 1643.19, 1469.28, 1504.77, 1528.02])
    V1 = np.array([1366.94, 1091.92, 997.93, 1013.47, 1104.05, 1387.08])
    U2 = np.array([1906.03, 1905.87, 1966.18, 1799.79, 1729.24, 1734.55])
    V2 = np.array([1351.90, 1077.88, 989.02, 975.90, 1061.55, 1337.05])

    # Hard code the location of these points in 3D space
    P1 = np.array([0, 0, 0])
    P2 = np.array([0, 0, 0.120])
    P3 = np.array([0.120, 0, 0.120])
    P4 = np.array([0.120, 0.073, 0.120])
    P5 = np.array([0, 0.073, 0.120])
    P6 = np.array([0, 0.073, 0])

    # Calculate intrinsic properties for both cameras
    M1, K1, R1, t1 = return_intrinsic_properties(P1, P2, P3, P4, P5, P6, U1, V1)
    M2, K2, R2, t2 = return_intrinsic_properties(P1, P2, P3, P4, P5, P6, U2, V2)

    # Calculate a 180 degree rotation around x-axis and apply it to R2
    R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R2 = R2 @ R_x

    # Define new and wall points in 2D
    UV1_NEW = np.array([[1968.05, 1559.88], [2202.76, 1220.26], [2187.66, 1424.09], 
                        [1980.20, 1324.09], [1125.65, 1722.11], [1323.53, 1589.71], [1986.17, 1128.66], 
                        [1215.53, 573.22], [2130.39, 1078.89], [2348.14, 1170.25], 
                        [2275.35, 1282.49], [2339.22, 1362.33]]) 
    UV2_NEW = np.array([[1458.19, 1525.89], [2331.92, 1294.59], [2307.73, 1504.32], 
                        [2202.91, 1369.19], [1144.02, 1544.47], [1398.63, 1473.58], [2218.13, 1168.49], 
                        [1318.51, 505.89], [2413.08, 1145.99], [2539.26, 1272.87],
                        [2431.68, 1374.62], [2522.58, 1479.28]])
    UV1_WALL = np.array([[2605.04, 694.93], [2367.25, 1126.04]])
    UV2_WALL = np.array([[3245.15, 833.20], [2745.68, 1246.69]])

    # Triangulate 3D points from 2D correspondences
    P_NEW = triangulate_points(M1, M2, UV1_NEW, UV2_NEW)
    P_WALL = triangulate_points(M1, M2, UV1_WALL, UV2_WALL)

    # Flatten initial points and M matrices for optimization
    initial_points = P_NEW.flatten()
    initial_M1 = M1.flatten()
    initial_M2 = M2.flatten()
    initial_params = np.hstack((initial_points, initial_M1, initial_M2))

    # Run the optimization with `minimize`
    result = minimize(
        bundle_adjustment_scalar,
        initial_params,
        args=(len(P_NEW), UV1_NEW, UV2_NEW),
        method='L-BFGS-B',
        options={'maxiter': 10000}  # Set higher for complex problems
    )

    # Extract optimized values
    optimized_points = result.x[:len(P_NEW) * 3].reshape(-1, 3)
    optimized_M1 = result.x[len(P_NEW) * 3 : len(P_NEW) * 3 + 12].reshape((3, 4))
    optimized_M2 = result.x[len(P_NEW) * 3 + 12 : len(P_NEW) * 3 + 24].reshape((3, 4))

    # Initialize the plot
    fig = go.Figure()

    # Plot initial 3D points (before refinement)
    x, y, z = P_NEW.T
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='blue'), name="Initial Points"))

    # Plot refined 3D points (after refinement)
    x_refined, y_refined, z_refined = optimized_points.T
    fig.add_trace(go.Scatter3d(x=x_refined, y=y_refined, z=z_refined, mode='markers', marker=dict(size=5, color='green'), name="Refined Points"))

    # Adding lines to show the shift from initial to refined positions
    for i in range(len(P_NEW)):
        fig.add_trace(go.Scatter3d(
            x=[P_NEW[i, 0], optimized_points[i, 0]],
            y=[P_NEW[i, 1], optimized_points[i, 1]],
            z=[P_NEW[i, 2], optimized_points[i, 2]],
            mode="lines",
            line=dict(color="gray", width=2),
            name="Refinement Path" if i == 0 else None  # Label only the first for legend clarity
        ))

    # Configure axes and layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="3D Points Before and After Refinement"
    )

    fig.show()

    # Translate all points so that they are relative to a new origin
    translation_vector = np.array([-0.634, -0.675, 0])
    P_NEW = optimized_points - translation_vector
    P_WALL = P_WALL - translation_vector
    P_CAL = np.array([P1, P2, P3, P4, P5, P6]) - translation_vector
    t1 = t1 - translation_vector
    t2 = t2 - translation_vector

    # Save the point clouds to a pickle file
    with open('point_clouds.pkl', 'wb') as f:
        pickle.dump({'P_CAL': P_CAL, 'P_NEW': P_NEW, 'P_WALL': P_WALL, 't1': t1, 't2': t2}, f)

    # Initialize Plotly figure
    fig = go.Figure()
    scale = 0.1

    # Plot the 3D points
    x, y, z = zip(*P_CAL)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='red'), name="Points (CAL)"))

    x, y, z = zip(*P_NEW)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='green'), name="Points (OBJS)"))

    x, y, z = zip(*P_WALL)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='blue'), name="Points (WALLS)"))

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
