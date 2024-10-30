clear;
clc;
close all;

% Load the images
imageFiles = {'image1.jpeg', 'image2.jpeg', 'image3.jpeg', 'image4.jpeg', 'image5.jpeg'}; % Add your actual image filenames
images = cell(1, numel(imageFiles));
for i = 1:numel(imageFiles)
    images{i} = imread(imageFiles{i});
end

% detectCheckerboardPointsCustom('image1.jpeg', 12, 0.8, 0.7);

% Detect the checkerboard corners
[imagePoints, boardSize] = detectCheckerboardPoints(imageFiles);

% Display the corners detected on the first image as a check
imshow(images{2});
hold on;
plot(imagePoints(:,1,2), imagePoints(:,2,2), 'ro'); % Red circles at the detected points
hold off;

% Define the size of the checkerboard squares (e.g., mm per square)
squareSize = 25;
worldPoints = patternWorldPoints("checkerboard",boardSize,squareSize);

% Using imagePoints (detected 2D points) and worldPoints (corresponding 3D points)
img = imread('image1.jpeg');
grayImage = rgb2gray(img);
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, 'ImageSize', size(grayImage));
