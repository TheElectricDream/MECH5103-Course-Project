clear;
clc;
close all;

load('cameraCalibrationResults_2024-10-17_rev1.mat');
load("selectedKeyPoints_2024-10-17_rev4.mat")

% Read stereo images
I1 = imread('image1.jpeg');
I2 = imread('image2.jpeg');

I1_GRAY = rgb2gray(I1);
I2_GRAY = rgb2gray(I2);

% Detect feature points in both images
points1 = detectBRISKFeatures(I1_GRAY);
points2 = detectBRISKFeatures(I2_GRAY);

% Extract feature descriptors
[features1, validPoints1] = extractFeatures(I1_GRAY, points1);
[features2, validPoints2] = extractFeatures(I2_GRAY, points2);

% Perform feature matching with Lowe's ratio test
indexPairs = matchFeatures(features1, features2, 'MaxRatio', 0.5, 'Unique', true, ...
    'Method','Exhaustive','MatchThreshold',10,'Metric','SAD');

% Retrieve matched points
matchedPoints1 = validPoints1(indexPairs(:, 1));
matchedPoints2 = validPoints2(indexPairs(:, 2));

% Visualize the matched points
figure; showMatchedFeatures(I1, I2, [u1 v1], [u2 v2]);

[F, inliersIndex] = estimateFundamentalMatrix([u1 v1], [u2 v2], Method="Norm8Point");

% Filter matched points based on inliers from RANSAC
inlierPoints1 = matchedPoints1(inliersIndex, :);
inlierPoints2 = matchedPoints2(inliersIndex, :);

% Visualize the inlier matches
figure; showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
title('Matched Inlier Points');

[E,inliers] = estimateEssentialMatrix([u1 v1], [u2 v2],cameraParams);

% % Extract the intrinsic matrix from cameraParams
% K = cameraParams.IntrinsicMatrix';
% 
% % Compute the essential matrix
% E = K' * F * K;

% Decompose the essential matrix to get the relative rotation and translation
%[R, t] = relativeCameraPose(E, cameraParams, inlierPoints1, inlierPoints2);
relPose = estrelpose(E,cameraParams.Intrinsics,[u1 v1], [u2 v2]);

R = relPose.R;
t = -relPose.Translation;

% Camera projection matrix for the first view (reference camera)
P1 = cameraMatrix(cameraParams, eye(3), [0 0 0]);

% Camera projection matrix for the second view (relative camera pose)
P2 = cameraMatrix(cameraParams, R, t);

% Triangulate the matched inlier points
points3D = triangulate([u1 v1], [u2 v2], P1, P2);

load('AdditionalKeyPoints.mat')

points3D_new = triangulate([u1, v1], [u2, v2], P1, P2);

% Visualize the 3D points
figure(1);
plot3(points3D(:,1), points3D(:,2), points3D(:,3), 'o');
hold on
plot3(points3D_new(:,1), points3D_new(:,2), points3D_new(:,3), 'x');
xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;
title('3D Reconstructed Points');

% Camera 1 (reference camera) is at the origin, with no rotation
camera1Position = [0 0 0]; % Camera 1 position at the origin
camera1Orientation = eye(3); % Camera 1 has no rotation (identity matrix)

% Camera 2 position and orientation based on relative pose (R, t)
camera2Position = -t * R'; % Convert the translation vector to world coordinates
camera2Orientation = R'; % Rotation matrix in world coordinates

% Create a figure for visualization
hold on;

% Plot Camera 1
plotCamera('Location', camera1Position, 'Orientation', camera1Orientation, ...
           'Size', 0.2, 'Color', 'b', 'Label', 'Camera 1', 'Opacity', 0.5);

% Plot Camera 2
plotCamera('Location', camera2Position, 'Orientation', camera2Orientation, ...
           'Size', 0.2, 'Color', 'r', 'Label', 'Camera 2', 'Opacity', 0.5);

% Add 3D plot settings
xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;
axis equal;
title('Camera Locations and Orientations');

%% UNCOMMENT TO SELECT NEW POINTS
% % Import images into MATLAB
% imageMatrix1 = imread('image1.jpeg', 'jpeg');
% imageMatrix2 = imread('image2.jpeg', 'jpeg');
% 
% % Plot image 1
% figure(1)
% imagesc(imageMatrix1)
% axis('equal')
% hold on
% 
% % Initialize arrays for selected points
% u1 = zeros(2, 1);
% v1 = zeros(2, 1);
% 
% % Select points from image 1
% for i = 1:2
%     zoom on
%     disp(['Zoom and press Enter to select point ', num2str(i), ' of ', num2str(2)]);
%     pause % Wait for the user to finish zooming and press Enter
%     zoom off % Disable zoom to allow point selection
%     [u1(i), v1(i)] = ginput(1); % Select one point at a time
%     scatter(u1(i), v1(i), 'r', 'filled'); % Plot the selected point
% end
% 
% % Plot image 2
% figure(2)
% imagesc(imageMatrix2)
% axis('equal')
% hold on
% 
% % Initialize arrays for selected points
% u2 = zeros(2, 1);
% v2 = zeros(2, 1);
% 
% % Select points from image 2
% for i = 1:2
%     zoom on
%     disp(['Zoom and press Enter to select point ', num2str(i), ' of ', num2str(2)]);
%     pause % Wait for the user to finish zooming and press Enter
%     zoom off % Disable zoom to allow point selection
%     [u2(i), v2(i)] = ginput(1); % Select one point at a time
%     scatter(u2(i), v2(i), 'r', 'filled'); % Plot the selected point
% end

