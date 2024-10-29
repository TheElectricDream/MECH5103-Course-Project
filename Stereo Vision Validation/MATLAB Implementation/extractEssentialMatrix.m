clear;
clc;
close all;

% Load the data
load('cameraFundamentalMatrix_2024-10-17_rev4.mat');
load("cameraCalibrationResults_2024-10-17_rev2.mat");
load("selectedKeyPoints_2024-10-17_rev4.mat");
%load('selectedKeyPoints_FeatureDetection_rev1.mat');

matchedPoints1.Location = [u1, v1];
matchedPoints2.Location = [u2, v2];

% Normalize intrinsic matrices
K1 = K1./K1(3,3);
K2 = K2./K2(3,3);
K1(1,2) = 0;
K2(1,2) = 0;

% Calculate Essential Matrix
E = transpose(K2) * (F * K1);

% Normalize E using SVD
[U, ~, V] = svd(E);

S = diag([1, 1, 0]);  % Force singular values

% Recalculate the essential Matrix:
E = U * S * transpose(V);

% Extract the translation of camera 2
t = U(:,3);

% Compute possible rotations and keep the two whose determinant == 1
counter = 1;
R_act = cell(4,1);
for i = 1:4
    if i == 1
        R = +U*transpose(rotz(+90))*transpose(V)*roty(+45);
    elseif i == 2
        R = -U*transpose(rotz(+90))*transpose(V)*roty(+45);
    elseif i == 3
        R = +U*transpose(rotz(-90))*transpose(V)*roty(+45);
    else 
        R = -U*transpose(rotz(-90))*transpose(V)*roty(+45);
    end
    
    if det(R) > 0.9
        R_act{counter} = R;
        counter = counter+1;
    end


end

% Generate all four possible solutions
poses = cell(4,1);
poses{1} = struct('R', R_act{1}, 't', t);
poses{2} = struct('R', R_act{1}, 't', -t);
poses{3} = struct('R', R_act{2}, 't', t);
poses{4} = struct('R', R_act{2}, 't', -t);

% Function to triangulate points
function X = triangulate_point(P1, P2, x1, x2)
    A = zeros(4,4);
    A(1,:) = x1(1) * P1(3,:) - P1(1,:);
    A(2,:) = x1(2) * P1(3,:) - P1(2,:);
    A(3,:) = x2(1) * P2(3,:) - P2(1,:);
    A(4,:) = x2(2) * P2(3,:) - P2(2,:);
    
    [~, ~, V] = svd(A);
    X = V(:,end);
    X = X(1:3) / X(4);
end

P1 = [eye(3) zeros(3,1)];
P2 = [poses{4}.R poses{4}.t];

% Convert image points to normalized coordinates
x1 = matchedPoints1.Location';
x2 = matchedPoints2.Location';

% Add homogeneous coordinates
x1 = [x1; ones(1, size(x1,2))];
x2 = [x2; ones(1, size(x2,2))];

% Normalize coordinates
x1_norm = K1 \ x1;
x2_norm = K2 \ x2;

for i = 1:size(x1,2)
    X(:,i) = triangulate_point(P1, P2, x1_norm(:,i), x2_norm(:,i));
end

% Create figure for visualization
figure('Position', [100, 100, 1200, 800]);
% Create subplot for 3D visualization
% Plot first camera
plotCamera('Location', [0 0 0], 'Orientation', eye(3), 'Size', 0.2, 'Color', 'b', 'Label', 'Camera 1');
hold on;

% Plot second camera
scale = 1; % Adjust this scale factor as needed
camera2_position = -scale * poses{4}.t;
plotCamera('Location', camera2_position', 'Orientation', poses{4}.R, 'Size', 0.2, 'Color', 'r', 'Label', 'Camera 2');

% Plot 3D points
% Scale points to match camera scale
points3D_scaled = scale * X;
scatter3(points3D_scaled(1,:), points3D_scaled(2,:), points3D_scaled(3,:), 20, 'g', 'filled');

% Add legend and labels
legend('Camera 1', 'Camera 2', '3D Points');
grid on;
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Camera Poses and 3D Points');
view(3);
