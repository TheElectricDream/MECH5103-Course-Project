import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt

# Disable scientific notation and set precision
np.set_printoptions(suppress=True,  # Suppress scientific notation
                   precision=4,      # Number of decimal places
                   floatmode='fixed')  # Fixed number of decimal places

def plot_epipolar_lines(F, U1, V1, U2, V2):

    # Plot the epipolar lines over the original images
    # Load the images
    img1 = cv2.imread('stereo_img1_small.jpg')
    img2 = cv2.imread('stereo_img2_small.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 8))
    img_plot = ax.imshow(img1)
    ax.axis('equal')

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    img_plot2 = ax2.imshow(img2)
    ax2.axis('equal')

    # Scatter plot the test points
    ax.scatter(U1, V1, c='r', marker='o')
    ax2.scatter(U2, V2, c='r', marker='o')

    # Run a loop to plot the epipolar lines
    for i in range(np.size(U1,0)):

        # Calculate the epipolar line
        L = F @ np.array([[U2[i]],[V2[i]],[1]])

        # Calculate the end points of the line
        x1 = 0
        y1 = np.round(-L[2]/L[1])
        x2 = img1.shape[1]
        y2 = np.round((-L[2] - L[0]*x2)/L[1])

        # Plot the line
        ax.plot([x1, x2], [y1, y2], 'g')

        # Calculate the epipolar line for the second image
        L = F.T @ np.array([[U1[i]],[V1[i]],[1]])

        # Calculate the end points of the line
        x1 = 0
        y1 = np.round(-L[2]/L[1])
        x2 = img1.shape[1]
        y2 = np.round((-L[2] - L[0]*x2)/L[1])

        # Plot the line
        ax2.plot([x1, x2], [y1, y2], 'g')

    # Make the axis tight to the size of the image
    ax.axis([0, img1.shape[1], img1.shape[0], 0])
    ax2.axis([0, img1.shape[1], img1.shape[0], 0])
    plt.show()

if __name__ == "__main__":

    # Hard code the points extract using the selectPointsForFundamentalMatrix.py script
    U1 = np.array([1025.97, 1147.92, 1471.66, 1726.62, 1967.05, 1920.81, 2308.22, 2414.17]) #, 1590.47, 1632.56])
    V1 = np.array([539.96, 1543.05, 1473.21, 1062.46, 988.92, 1386.99, 1503.24, 1145.77]) #, 1595.54, 2033.59])
    U2 = np.array([1022.68, 1132.70, 1429.26, 1770.82, 2067.17, 1927.17, 2177.51, 2419.28]) #, 985.09, 1168.71])
    V2 = np.array([626.06, 1539.33, 1526.08, 1174.05, 1133.43, 1496.75, 1666.24, 1330.60]) #, 1579.98, 2021.77])

    # Calculate number of points
    N_PTS = np.size(U1,0)

    # Calculate the means
    U1_AVG = np.average(U1)
    U2_AVG = np.average(U2)
    V1_AVG = np.average(V1)
    V2_AVG = np.average(V2)

    # Calculate mean distances
    MD_1 = (1/N_PTS)*np.sum(np.sqrt((U1-U1_AVG)**2 + (V1-V1_AVG)**2))
    MD_2 = (1/N_PTS)*np.sum(np.sqrt((U2-U2_AVG)**2 + (V2-V2_AVG)**2))

    # Calculate the normlization scale factors
    S1 = np.sqrt(2) / MD_1
    S2 = np.sqrt(2) / MD_2

    # Assemble the normalization matrices
    GAMMA_1 = np.array([[S1, 0, -S1*U1_AVG], [0, S1, -S1*V1_AVG], [0, 0, 1]])
    GAMMA_2 = np.array([[S2, 0, -S2*U2_AVG], [0, S2, -S2*V2_AVG], [0, 0, 1]])
    
    # Initialize the dicts
    point_dict_1 = {}
    point_dict_2 = {}

    # Assemble point vectors and normalize
    for i in range(np.size(U1,0)):
        point_dict_1[f'P{i+1}_NORM'] = GAMMA_1 @ np.array([[U1[i]],[V1[i]],[1]])
        point_dict_2[f'P{i+1}_NORM'] = GAMMA_2 @ np.array([[U2[i]],[V2[i]],[1]])

    # Reassemble the point vectors into one array using hstack
    P1_ARR = np.array([point_dict_1[f'P{i+1}_NORM'] for i in range(np.size(U1, 0))])
    P1_ARR = P1_ARR.T

    # Flatten array into 3,8
    P1_ARR = P1_ARR.reshape(3,8)

    P2_ARR = np.array([point_dict_2[f'P{i+1}_NORM'] for i in range(np.size(U2, 0))])
    P2_ARR = P2_ARR.T   
    
    # Flatten array into 3,8
    P2_ARR = P2_ARR.reshape(3,8)

    # Assemble the Y matrix
    for i in range(np.size(U1,0)):

        Y11 = P1_ARR[0,i]*P2_ARR[0,i]
        Y12 = P1_ARR[0,i]*P2_ARR[1,i]
        Y13 = P1_ARR[0,i]

        Y21 = P1_ARR[1,i]*P2_ARR[0,i]
        Y22 = P1_ARR[1,i]*P2_ARR[1,i]
        Y23 = P1_ARR[1,i]

        Y31 = P2_ARR[0,i]
        Y32 = P2_ARR[1,i]
        Y33 = 1

        if i == 0:
            Y = np.array([Y11, Y12, Y13, Y21, Y22, Y23, Y31, Y32, Y33])
        else:
            Y = np.vstack((Y, np.array([Y11, Y12, Y13, Y21, Y22, Y23, Y31, Y32, Y33])))

    # Calculate the SVD of Y
    UY_SVD, SY_SVD, VHY_SVD = np.linalg.svd(Y)

    # Transpose V
    VY_SVD = VHY_SVD.T

    # Extract the normalized F matrix
    F_NORM = np.array([[VY_SVD[0,-1], VY_SVD[1,-1], VY_SVD[2,-1]], [VY_SVD[3,-1], VY_SVD[4,-1], VY_SVD[5,-1]], [VY_SVD[6,-1], VY_SVD[7,-1], VY_SVD[8,-1]]])

    # Enforce rank 2 constraint on F
    UF_SVD, SF_SVD, VHF_SVD = np.linalg.svd(F_NORM)
    SF_SVD[-1] = 0
    SF_SVD = np.diag(SF_SVD)
    F_NORM = UF_SVD @ SF_SVD @ VHF_SVD

    # Denormalize F
    F = GAMMA_1.T @ F_NORM @ GAMMA_2

    # Save the F matrix
    with open('fundamental_matrix.pkl', 'wb') as f:
        pickle.dump(F, f)

    # Plot the epipolar lines
    plot_epipolar_lines(F, U1, V1, U2, V2)
