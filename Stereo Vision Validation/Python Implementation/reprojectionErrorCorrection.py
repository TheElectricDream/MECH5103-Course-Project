import cv2
import numpy as np

M1 = np.array([[ 0.2455,  0.5428, -0.0074,  0.4766],
       [ 0.0901, -0.0752, -0.5500,  0.3187],
       [-0.0001,  0.0001, -0.0000,  0.0002]])

M2 = np.array([[ 0.0796, -0.5463, -0.0686, -0.5352],
       [-0.0802,  0.0678,  0.5320, -0.3316],
       [ 0.0001,  0.0000, -0.0000, -0.0002]])

object_points = np.array([[ 0.1285, -0.0010,  0.0004],
       [ 0.1280,  0.0879,  0.0004],
       [ 0.1275,  0.0881,  0.0853],
       [-0.0007,  0.0870,  0.0870],
       [-0.0004,  0.0009,  0.0872],
       [ 0.0003, -0.0001, -0.0006],
       [ 0.1277, -0.0012,  0.0002],
       [ 0.1302,  0.0005,  0.0885],
       [-0.0014,  0.0013,  0.0868],
       [ 0.1299,  0.0008,  0.0881],
       [ 0.1320,  0.0432,  0.0506],
       [ 0.1285,  0.0871,  0.0862]])

image_points1 = np.array([[2188.3700, 1423.9500],
       [2339.9100, 1362.1100],
       [2348.8800, 1170.9100],
       [2132.1900, 1077.4400],
       [1985.3700, 1128.4300],
       [1976.8700, 1323.4000],
       [2186.4800, 1423.4800],
       [2203.0000, 1221.4300],
       [1984.4300, 1128.4300],
       [2203.0000, 1221.9000],
       [2276.6500, 1284.2200],
       [2348.8800, 1170.4400]])

image_points2 = np.array([[2307.7200, 1503.1200],
       [2523.7300, 1478.2700],
       [2540.9100, 1274.4500],
       [2413.9200, 1145.2000],
       [2219.1400, 1167.3500],
       [2199.7100, 1364.8300],
       [2306.3600, 1503.1200],
       [2331.2200, 1294.3400],
       [2219.1400, 1167.3500],
       [2331.6700, 1294.7900],
       [2429.2800, 1372.0700],
       [2539.5500, 1273.1000]])

image_size = (4032, 3024)
# Assume you have these variables already
# image_points1: Nx2 array of 2D points in image from camera 1
# image_points2: Nx2 array of 2D points in image from camera 2
# object_points: Nx3 array of triangulated 3D points
# image_size: (width, height) of the images

# Convert points to the required format
object_points_list = [object_points.astype(np.float32)]
image_points1_list = [image_points1.astype(np.float32)]
image_points2_list = [image_points2.astype(np.float32)]

# Decompose initial projection matrices M1 and M2 to get initial camera matrices and poses
def decompose_projection_matrix(M):
    K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(M)
    K = K / K[2,2]  # Normalize so that K[2,2] = 1
    t = (t / t[3])[:3]  # Convert to non-homogeneous coordinates
    return K, R, t

# Decompose M1 and M2
K1_init, R1_init, t1_init = decompose_projection_matrix(M1)
K2_init, R2_init, t2_init = decompose_projection_matrix(M2)

# Convert rotation matrices to rotation vectors
rvec1_init, _ = cv2.Rodrigues(R1_init)
rvec2_init, _ = cv2.Rodrigues(R2_init)

# Set initial distortion coefficients to zero
dist_coeffs1 = np.zeros((5, 1), dtype=np.float64)
dist_coeffs2 = np.zeros((5, 1), dtype=np.float64)

# Stereo calibration flags
flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
         cv2.CALIB_FIX_PRINCIPAL_POINT +
         cv2.CALIB_FIX_FOCAL_LENGTH +
         cv2.CALIB_ZERO_TANGENT_DIST +
         cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 +
         cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 +
         cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6)

# Stereo calibration criteria
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# Run stereo calibration to refine camera parameters
ret, K1_refined, dist_coeffs1_refined, K2_refined, dist_coeffs2_refined, R, T, E, F = cv2.stereoCalibrate(
    object_points_list, image_points1_list, image_points2_list,
    K1_init, dist_coeffs1, K2_init, dist_coeffs2,
    image_size, criteria=criteria, flags=flags)

# Compute reprojection error for camera 1
def compute_reprojection_error(object_points, image_points, rvec, tvec, K, dist_coeffs):
    projected_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, K, dist_coeffs)
    error = cv2.norm(image_points, projected_points.reshape(-1, 2), cv2.NORM_L2)
    return error / len(object_points)

# Update rotation and translation vectors
rvecs1 = [np.zeros((3, 1), dtype=np.float64)]  # Camera 1 is the reference
tvecs1 = [np.zeros((3, 1), dtype=np.float64)]
rvecs2 = [cv2.Rodrigues(R)[0]]
tvecs2 = [T]

# Compute errors
error1 = compute_reprojection_error(
    object_points_list[0], image_points1_list[0], rvecs1[0], tvecs1[0],
    K1_refined, dist_coeffs1_refined)
error2 = compute_reprojection_error(
    object_points_list[0], image_points2_list[0], rvecs2[0], tvecs2[0],
    K2_refined, dist_coeffs2_refined)

print(f"Reprojection error for camera 1: {error1}")
print(f"Reprojection error for camera 2: {error2}")
print(f"Mean reprojection error: {(error1 + error2) / 2}")

# The refined projection matrices
P1_refined = K1_refined @ np.hstack((np.eye(3), np.zeros((3, 1))))
R2, _ = cv2.Rodrigues(rvecs2[0])
P2_refined = K2_refined @ np.hstack((R2, T))

print("Refined Projection Matrix for Camera 1:")
print(P1_refined)
print("\nRefined Projection Matrix for Camera 2:")
print(P2_refined)