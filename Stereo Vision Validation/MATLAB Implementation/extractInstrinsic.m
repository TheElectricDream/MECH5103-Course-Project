clear;
clc;
close all;

%% EXTRACT PHYSICAL LOCATION OF POINTS
% p1 = [0, 0, 0];
% p2 = [0.225, 0, 0];
% p3 = [0.225, 0, 0.070];
% p4 = [0.225, 0.122, 0.070];
% p5 = [0, 0.122, 0.070];
% p6 = [0, 0, 0.070];

%% UNCOMMENT TO SELECT NEW POINTS
% % Import images into MATLAB
% imageMatrix1 = imread('ppmcal_img1.jpeg','jpeg');
% imageMatrix2 = imread('ppmcal_img2.jpeg','jpeg');
% 
% % Plot image 1
% figure(1)
% imagesc(imageMatrix1)
% axis('equal')
% hold on
% 
% % Select points from image 1
% [u1,v1] = ginput(6);
% 
% scatter(u1,v1);
% 
% % Plot image 2
% figure(2)
% imagesc(imageMatrix2)
% axis('equal')
% hold on
% 
% % Select points from image 2
% [u2,v2] = ginput(6);
% 
% scatter(u2,v2);
% 
%

% These are the points in pixels
% u1 = [1.426866960853654e+03;2.939601595990037e+03;3.085115148067284e+03;2.657669088840370e+03;1.199502035732955e+03;1.405646234509056e+03];
% v1 = [2.058175820289677e+03;1.694391940096559e+03;1.345765721578154e+03;8.394998216427320e+02;1.057770149758603e+03;1.697423472431502e+03];
% 
% % These are the equivalent points in the second image
% u2 = [1.351078652480088e+03;2.484871745748639e+03;2.530344730772779e+03;2.933538531320151e+03;1.778524711707002e+03;1.299542602786063e+03];
% v2 = [1.467027014975860e+03;2.085459611304161e+03;1.755022586795412e+03;1.291198139549187e+03;8.273736923029614e+02;1.139621522802054e+03];

% u1 = [1353.68, 2485.49, 2531.69, 2933.59, 1778.69, 1295.94];
% v1 = [1461.17, 2082.51, 1749.90, 1285.62, 819.04, 1126.25];
% u2 = [1423.72, 2928.60, 3087.15, 2658.78, 1203.97, 1404.25];
% v2 = [2050.92, 1694.87, 1338.82, 835.34, 1049.53, 1686.53];

u1 = [1906.76, 1967.07, 1965.47, 1799.44, 1727.68, 1906.09];
v1 = [1357.04, 1245.04, 988.24, 976.21, 1063.03, 1078.09];
u2 = [1916.95, 2065.50, 2065.89, 1922.46, 1771.20, 1917.03];
v2 = [1479.07, 1383.47, 1132.67, 1107.04, 1174.32, 1205.45];

p1 = [0, 0, 0];
p2 = [0.120, 0, 0];
p3 = [0.120, 0, 0.120];
p4 = [0.120, 0.073, 0.120];
p5 = [0, 0.073, 0.120];
p6 = [0, 0, 0.120];


%% CALCULATE PARAMETERS FOR CAMERA 1
A1 = zeros(12,12);
A1(1,:) = [p1, 1, 0, 0 ,0 ,0, -u1(1)*p1(1), -u1(1)*p1(2), -u1(1)*p1(3), -u1(1)];
A1(2,:) = [0, 0, 0, 0, p1, 1, -v1(1)*p1(1), -v1(1)*p1(2), -v1(1)*p1(3), -v1(1)];

A1(3,:) = [p2, 1, 0, 0 ,0 ,0, -u1(2)*p2(1), -u1(2)*p2(2), -u1(2)*p2(3), -u1(2)];
A1(4,:) = [0, 0, 0, 0, p2, 1, -v1(2)*p2(1), -v1(2)*p2(2), -v1(2)*p2(3), -v1(2)];

A1(5,:) = [p3, 1, 0, 0 ,0 ,0, -u1(3)*p3(1), -u1(3)*p3(2), -u1(3)*p3(3), -u1(3)];
A1(6,:) = [0, 0, 0, 0, p3, 1, -v1(3)*p3(1), -v1(3)*p3(2), -v1(3)*p3(3), -v1(3)];

A1(7,:) = [p4, 1, 0, 0 ,0 ,0, -u1(4)*p4(1), -u1(4)*p4(2), -u1(4)*p4(3), -u1(4)];
A1(8,:) = [0, 0, 0, 0, p4, 1, -v1(4)*p4(1), -v1(4)*p4(2), -v1(4)*p4(3), -v1(4)];

A1(9,:) = [p5, 1, 0, 0 ,0 ,0, -u1(5)*p5(1), -u1(5)*p5(2), -u1(5)*p5(3), -u1(5)];
A1(10,:) = [0, 0, 0, 0, p5, 1, -v1(5)*p5(1), -v1(5)*p5(2), -v1(5)*p5(3), -v1(5)];

A1(11,:) = [p6, 1, 0, 0 ,0 ,0, -u1(6)*p6(1), -u1(6)*p6(2), -u1(6)*p6(3), -u1(6)];
A1(12,:) = [0, 0, 0, 0, p6, 1, -v1(6)*p6(1), -v1(6)*p6(2), -v1(6)*p6(3), -v1(6)];

% Calculate the SVD

[U1, S1, V1] = svd(A1);

% Assemble the M matrix

M1 = [V1(1,end),V1(2,end),V1(3,end),V1(4,end)
      V1(5,end),V1(6,end),V1(7,end),V1(8,end)
      V1(9,end),V1(10,end),V1(11,end),V1(12,end)];

% Extract the camera parameters using QR decomposition

M1_3x3 = M1(1:3, 1:3);
[R1_inv, K1_inv] = qr(flipud(M1_3x3)');  % QR decomposition on the flipped matrix
R1 = flipud(R1_inv');
K1 = rot90(K1_inv', 2);  % Correct orientation for K

% Correct sign of K

T1 = diag(sign(diag(K1)));
K1 = K1 * T1;
R1 = T1 * R1;

% Calculate the position (NOTE THERE IS SOMETHING WEIRD WITH THE UNITS)

o1 = -M1\[M1(:,4)];

%% CALCULATE PARAMETERS FOR CAMERA 2
A2 = zeros(12,12);
A2(1,:) = [p1, 1, 0, 0 ,0 ,0, -u2(1)*p1(1), -u2(1)*p1(2), -u2(1)*p1(3), -u2(1)];
A2(2,:) = [0, 0, 0, 0, p1, 1, -v2(1)*p1(1), -v2(1)*p1(2), -v2(1)*p1(3), -v2(1)];

A2(3,:) = [p2, 1, 0, 0 ,0 ,0, -u2(2)*p2(1), -u2(2)*p2(2), -u2(2)*p2(3), -u2(2)];
A2(4,:) = [0, 0, 0, 0, p2, 1, -v2(2)*p2(1), -v2(2)*p2(2), -v2(2)*p2(3), -v2(2)];

A2(5,:) = [p3, 1, 0, 0 ,0 ,0, -u2(3)*p3(1), -u2(3)*p3(2), -u2(3)*p3(3), -u2(3)];
A2(6,:) = [0, 0, 0, 0, p3, 1, -v2(3)*p3(1), -v2(3)*p3(2), -v2(3)*p3(3), -v2(3)];

A2(7,:) = [p4, 1, 0, 0 ,0 ,0, -u2(4)*p4(1), -u2(4)*p4(2), -u2(4)*p4(3), -u2(4)];
A2(8,:) = [0, 0, 0, 0, p4, 1, -v2(4)*p4(1), -v2(4)*p4(2), -v2(4)*p4(3), -v2(4)];

A2(9,:) = [p5, 1, 0, 0 ,0 ,0, -u2(5)*p5(1), -u2(5)*p5(2), -u2(5)*p5(3), -u2(5)];
A2(10,:) = [0, 0, 0, 0, p5, 1, -v2(5)*p5(1), -v2(5)*p5(2), -v2(5)*p5(3), -v2(5)];

A2(11,:) = [p6, 1, 0, 0 ,0 ,0, -u2(6)*p6(1), -u2(6)*p6(2), -u2(6)*p6(3), -u2(6)];
A2(12,:) = [0, 0, 0, 0, p6, 1, -v2(6)*p6(1), -v2(6)*p6(2), -v2(6)*p6(3), -v2(6)];

% Calculate the SVD

[U2, S2, V2] = svd(A2);

% Assemble the M matrix

M2 = [V2(1,end),V2(2,end),V2(3,end),V2(4,end)
      V2(5,end),V2(6,end),V2(7,end),V2(8,end)
      V2(9,end),V2(10,end),V2(11,end),V2(12,end)];

% Extract the camera parameters using QR decomposition

M2_3x3 = M2(1:3, 1:3);
[R2_inv, K2_inv] = qr(flipud(M2_3x3)');  % QR decomposition on the flipped matrix
R2 = flipud(R2_inv');
K2 = rot90(K2_inv', 2);  % Correct orientation for K

% Correct sign of K

T2 = diag(sign(diag(K2)));
K2 = K2 * T2;
R2 = T2 * R2;

% Calculate the position (NOTE THERE IS SOMETHING WEIRD WITH THE UNITS)

o2 = -M2\[M2(:,4)];

%% PLOT THE POINTS AND THE CAMERAS
figure()
scatter3(p1(1),p1(2),p1(3))  % Point 1
hold on
scatter3(p2(1),p2(2),p2(3))  % Point 2
scatter3(p3(1),p3(2),p3(3))  % Point 3
scatter3(p4(1),p4(2),p4(3))  % Point 4
scatter3(p5(1),p5(2),p5(3))  % Point 5
scatter3(p6(1),p6(2),p6(3))  % Point 6
scatter3(o1(1),o1(2),o1(3), 'filled')  % Camera 1 position
scatter3(o2(1),o2(2),o2(3), 'filled')  % Camera 2 position

% Optional: Add labels and grid for better visualization
xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;
title('3D Points and Camera Position');

% Set up the camera's local coordinate axes
axis_length = 0.1;  % Adjust this scale for visibility

% Extract the camera's local X, Y, and Z axes from the rotation matrix
x_axis = R1(:, 1) * axis_length;  % Camera X-axis
y_axis = R1(:, 2) * axis_length;  % Camera Y-axis
z_axis = R1(:, 3) * axis_length;  % Camera Z-axis

% Plot the X-axis (in red)
quiver3(o1(1), o1(2), o1(3), x_axis(1), x_axis(2), x_axis(3), 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);

% Plot the Y-axis (in green)
quiver3(o1(1), o1(2), o1(3), y_axis(1), y_axis(2), y_axis(3), 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);

% Plot the Z-axis (in blue)
quiver3(o1(1), o1(2), o1(3), z_axis(1), z_axis(2), z_axis(3), 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);

% Extract the camera's local X, Y, and Z axes from the rotation matrix
x_axis = R2(:, 1) * axis_length;  % Camera X-axis
y_axis = R2(:, 2) * axis_length;  % Camera Y-axis
z_axis = R2(:, 3) * axis_length;  % Camera Z-axis

% Plot the X-axis (in red)
quiver3(o2(1), o2(2), o2(3), x_axis(1), x_axis(2), x_axis(3), 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);

% Plot the Y-axis (in green)
quiver3(o2(1), o2(2), o2(3), y_axis(1), y_axis(2), y_axis(3), 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);

% Plot the Z-axis (in blue)
quiver3(o2(1), o2(2), o2(3), z_axis(1), z_axis(2), z_axis(3), 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);

axis square