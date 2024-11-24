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