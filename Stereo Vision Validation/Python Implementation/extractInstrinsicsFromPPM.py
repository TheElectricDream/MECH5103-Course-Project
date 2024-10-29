import numpy as np
import pickle

# Disable scientific notation and set precision
np.set_printoptions(suppress=True,  # Suppress scientific notation
                   precision=4,      # Number of decimal places
                   floatmode='fixed')  # Fixed number of decimal places

def return_intrinsic_properties(P1, P2, P3, P4, P5, P6, U, V):

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

    # Extract the rotation and translation components using RQ decomposition
    M_3X3 = M[:3,:3]
    R_inv, K_inv = np.linalg.qr(np.transpose(np.flipud(M_3X3)))
    R = np.flipud(np.transpose(R_inv))
    K = np.rot90(np.transpose(K_inv), 2)

    # Enforce positive diagonals in K
    K = np.sign(K)*K

    # Normalize K so that element [3,3] is equal to 1
    K = K/K[-1,-1]

    # Set the skew element to 0
    K[0,1] = 0

    return K, M, A


# Main function
if __name__ == "__main__":

    # Hard code the points extract using the selectPointsForPPM.py script
    U1 = np.array([1353.68, 2485.49, 2531.69, 2933.59, 1778.69, 1295.94])
    V1 = np.array([1461.17, 2082.51, 1749.90, 1285.62, 819.04, 1126.25])
    U2 = np.array([1423.72, 2928.60, 3087.15, 2658.78, 1203.97, 1404.25])
    V2 = np.array([2050.92, 1694.87, 1338.82, 835.34, 1049.53, 1686.53])

    # Hard code the location of these points in 3D space
    P1 = np.array([0, 0, 0])
    P2 = np.array([0.225, 0, 0])
    P3 = np.array([0.225, 0, 0.070])
    P4 = np.array([0.225, 0.122, 0.070])
    P5 = np.array([0, 0.122, 0.070])
    P6 = np.array([0, 0, 0.070])

    K1, M1, A1 = return_intrinsic_properties(P1, P2, P3, P4, P5, P6, U1, V1)
    K2, M2, A2 = return_intrinsic_properties(P1, P2, P3, P4, P5, P6, U2, V2)

    # Save all calibration data
    calibration_data = {
        'M1': M1,
        'M2': M2,
        'K1': K1,
        'K2': K2,
        'A1': A1,
        'A2': A2,
        'points': {
            'P1': P1,
            'P2': P2,
            'P3': P3,
            'P4': P4,
            'P5': P5,
            'P6': P6
        },
        'image_points': {
            'U1': U1,
            'V1': V1,
            'U2': U2,
            'V2': V2
        }
    }

    # Save calibration data to a pickle file with a time stamp in the filename
    with open(f'calibration_data.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)

    print('Calibration data saved to calibration_data.pkl')