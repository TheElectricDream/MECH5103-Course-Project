% Clear old variables
clear;
clc;
close all;

% Define the total number of points
nPointsTotal = 10;
nPointsTest = 8;

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
% u1 = zeros(nPointsTotal, 1);
% v1 = zeros(nPointsTotal, 1);
% 
% % Select points from image 1
% for i = 1:nPointsTotal
%     zoom on
%     disp(['Zoom and press Enter to select point ', num2str(i), ' of ', num2str(nPointsTotal)]);
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
% u2 = zeros(nPointsTotal, 1);
% v2 = zeros(nPointsTotal, 1);
% 
% % Select points from image 2
% for i = 1:nPointsTotal
%     zoom on
%     disp(['Zoom and press Enter to select point ', num2str(i), ' of ', num2str(nPointsTotal)]);
%     pause % Wait for the user to finish zooming and press Enter
%     zoom off % Disable zoom to allow point selection
%     [u2(i), v2(i)] = ginput(1); % Select one point at a time
%     scatter(u2(i), v2(i), 'r', 'filled'); % Plot the selected point
% end


%% START FROM SAVED POINTS
% Load the data that was saved containing the points
load('selectedKeyPoints_2024-10-17_rev4.mat');

% Recompute the fundamental matrix from matched points
%[F_MATLAB, inliers] = estimateFundamentalMatrix([u1,v1], [u2,v2], 'Method', 'RANSAC');

% Display the new fundamental matrix
% disp('Recomputed Fundamental Matrix F:');
% disp(F);

u1 = [1025.97, 1147.92, 1471.66, 1726.62, 1967.05, 1920.81, 2308.22, 2414.17, 1590.47, 1632.56];
v1 = [539.96, 1543.05, 1473.21, 1062.46, 988.92, 1386.99, 1503.24, 1145.77, 1595.54, 2033.59];
u2 = [1022.68, 1132.70, 1429.26, 1770.82, 2067.17, 1927.17, 2177.51, 2419.28, 985.09, 1168.71];
v2 = [626.06, 1539.33, 1526.08, 1174.05, 1133.43, 1496.75, 1666.24, 1330.60, 1579.98, 2021.77];


% Create the point cell arrays
for i = 1:nPointsTest

    img1_cells{i} = [u1(i);v1(i);1];
    img2_cells{i} = [u2(i);v2(i);1];

end

% Calculate the centroids of the test points
u1_mean = (1/8)*sum(u1(1:nPointsTest));
v1_mean = (1/8)*sum(v1(1:nPointsTest));
u2_mean = (1/8)*sum(u2(1:nPointsTest));
v2_mean = (1/8)*sum(v2(1:nPointsTest));

% Calculate the mean distances
mean_distance1 = (1/8)*sum(sqrt((u1(1:nPointsTest)-u1_mean).^2 + (v1(1:nPointsTest)-v1_mean).^2));
mean_distance2 = (1/8)*sum(sqrt((u2(1:nPointsTest)-u2_mean).^2 + (v2(1:nPointsTest)-v2_mean).^2));

% Calculate the normalization scale factors
s1 = sqrt(2) / mean_distance1;
s2 = sqrt(2) / mean_distance2;

% Calculate the normalization matrices
Gamma1 = [s1 0 -s1*u1_mean; 0 s1 -s1*v1_mean; 0 0 1];
Gamma2 = [s2 0 -s2*u2_mean; 0 s2 -s2*v2_mean; 0 0 1];

% Normalize the points
for i = 1:nPointsTest

    img1_cells_norm{i} = Gamma1*img1_cells{i};
    img2_cells_norm{i} = Gamma2*img2_cells{i};

end

% Create the normalized matrices from the cell arrays
img1_mat_norm = cell2mat(img1_cells_norm);
img2_mat_norm = cell2mat(img2_cells_norm);

% Append the [1;1;1] at the end of the point matrices
img1_mat_norm = horzcat(img1_mat_norm,[1;1;1]);
img2_mat_norm = horzcat(img2_mat_norm,[1;1;1]);

% Calculate the normalized Y matrix
Yn = zeros(nPointsTest,9);

for i = 1:nPointsTest

    Y11 = img1_mat_norm(1,i)*img2_mat_norm(1,i);
    Y12 = img1_mat_norm(1,i)*img2_mat_norm(2,i);
    Y13 = img1_mat_norm(1,i);

    Y14 = img1_mat_norm(2,i)*img2_mat_norm(1,i);
    Y15 = img1_mat_norm(2,i)*img2_mat_norm(2,i);
    Y16 = img1_mat_norm(2,i);

    Y17 = img2_mat_norm(1,i);
    Y18 = img2_mat_norm(2,i);
    Y19 = 1;

    Yn(i,:) = [Y11, Y12, Y13, Y14, Y15, Y16, Y17, Y18, Y19];

end

% Extract the SVD components of Yn
[Uy,Sy,Vy] = svd(Yn);

% Extract the normalized F matrix
FnVec = Vy(:,end);
Fn = [FnVec(1),FnVec(2),FnVec(3);FnVec(4),FnVec(5),FnVec(6);FnVec(7),FnVec(8),FnVec(9)];

% Compute the smallest singular value of F, and zero it
[Uf, Sf, Vf] = svd(Fn);
SfStar = Sf;
SfStar(3,3) = 0;

% Calculate the singular normalized F matrix
FnStar = Uf*SfStar*transpose(Vf);

% Calculate the original F matrix
F = transpose(Gamma1)*FnStar*Gamma2;

%% PLOT THE RESULTS
% Import images into MATLAB
imageMatrix1 = imread('image1.jpeg','jpeg');
imageMatrix2 = imread('image2.jpeg','jpeg');

% Plot the images
figure(1)
imagesc(imageMatrix1)
hold on

figure(2)
imagesc(imageMatrix2)
hold on

% Re-load the image cell arrays so they contain the non-test points
for i = 1:nPointsTotal

    img1_cells{i} = [u1(i);v1(i);1];
    img2_cells{i} = [u2(i);v2(i);1];

end

% Run a loop to plot the epipolar lines
for i = 1:nPointsTotal

    % Calculate the vector for one point on each figure
    p1 = img1_cells{i};
    p2 = img2_cells{i};

    a2 = transpose(F)*p1;
    a1 = F*p2;

    % Plot image 1
    figure(1)
    if i <= 8
        scatter(p1(1), p1(2), 'red', 'filled');
    else
        scatter(p1(1), p1(2), 'blue', 'filled');
    end
    u1_pt = 0;
    u2_pt = 4000;
    v1_pt = -(a1(1)/a1(2))*u1_pt - a1(3)/(a1(2));
    v2_pt = -(a1(1)/a1(2))*u2_pt - a1(3)/(a1(2));
    line([u1_pt, u2_pt],[v1_pt, v2_pt])
    axis([0 4000 0 3000])

    % Plot image 2
    figure(2)
    if i <= 8
        scatter(p2(1), p2(2), 'red', 'filled');
    else
        scatter(p2(1), p2(2), 'blue', 'filled');
    end
    u1_pt = 0;
    u2_pt = 4000;
    v1_pt = -(a2(1)/a2(2))*u1_pt - a2(3)/(a2(2));
    v2_pt = -(a2(1)/a2(2))*u2_pt - a2(3)/(a2(2));
    line([u1_pt, u2_pt],[v1_pt, v2_pt])
    axis([0 4000 0 3000])

end

% % Add a legend
% figure(1)
% legend('','','','','','','Test Point','','Non-Test Point','','','','','','','',...
%     '','','','','Location','best')
% exportgraphics(gcf, 'image1_new.jpeg', 'Resolution', 300); % 300 DPI for higher quality
% 
% figure(2)
% legend('','','','','','','Test Point','','Non-Test Point','','','','','',...
%     '','','','','','','Location','best')
% exportgraphics(gcf, 'image3_new.jpeg', 'Resolution', 300); % 300 DPI for higher quality