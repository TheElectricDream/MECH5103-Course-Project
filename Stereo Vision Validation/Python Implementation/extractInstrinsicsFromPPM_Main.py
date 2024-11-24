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

# Main function
if __name__ == "__main__":

    # Hard code the points extracted using the selectPointsForPPM.py script
    # These points are for revFinal
    U1 = np.array([994.04, 1130.81, 1323.45, 1527.57, 1643.08, 1984.40, 1706.82, 1691.76, 2201.92, 2348.01])
    V1 = np.array([639.72, 1722.25, 1589.26, 1387.17, 997.37, 1129.09, 1365.43,1092.20, 1218.78, 1168.31])
    U2 = np.array([1030.41, 1149.69, 1398.14, 1734.31, 1966.41, 2217.97,1905.45,1904.88, 2331.52, 2538.64])
    V2 = np.array([540.94, 1548.72, 1473.72, 1337.16, 989.52, 1169.19,1350.05,1078.66, 1293.54, 1273.36])

    # These points are for revFinal
    P1 = np.array([0.42409, 0.88457, 0.37098])
    P2 = np.array([0.42409, 0.88457, 0])
    P3 = np.array([0.49632, 0.82061, 0])
    P4 = np.array([0.63478, 0.73245, 0])
    P5 = np.array([0.75478, 0.65945, 0.120])
    P6 = np.array([0.65714, 0.53881, 0.087])
    P7 = np.array([0.63478, 0.65945, 0])
    P8 = np.array([0.63478, 0.65945, 0.120])
    P9 = np.array([0.541128, 0.484716, 0.087])
    P10 = np.array([0.577896, 0.405867, 0.087])

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

    # Triangulate 3D points from 2D correspondences
    P_WOOD    = triangulate_points(M1, M2, UV1_WOOD, UV2_WOOD)
    P_TISSUE  = triangulate_points(M1, M2, UV1_TISSUE, UV2_TISSUE)
    P_BOX     = triangulate_points(M1, M2, UV1_BOX, UV2_BOX)
    P_CAL     = np.array([P1, P2, P3, P4, P5, P6, P7, P8])

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
