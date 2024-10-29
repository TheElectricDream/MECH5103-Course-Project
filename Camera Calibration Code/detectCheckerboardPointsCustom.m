function [] = detectCheckerboardPointsCustom(imageFile, sigma, highThresholdValue, lowThresholdValue)

    % Load in the image
    img = imread(imageFile);
    grayImage = rgb2gray(img);

    % Do the edge dectection
    % Start with a Guassian filter to smooth out any noise in the image
    filterSize = 2 * ceil(3 * sigma) + 1;

    % Manually create the Gaussian kernel
    [x, y] = meshgrid(-ceil(3 * sigma):ceil(3 * sigma), -ceil(3 * sigma):ceil(3 * sigma));
    G = exp(-(x.^2 + y.^2) / (2 * sigma^2)) / (2 * pi * sigma^2);
    G = G / sum(G(:));  % Normalize the kernel

    % Apply Gaussian filter to smooth the image
    smoothedImage = imfilter(grayImage, G, 'same');

    % Sobel kernels to compute gradients in x and y directions
    SobelX = [-1 0 1; -2 0 2; -1 0 1];
    SobelY = [-1 -2 -1; 0 0 0; 1 2 1];
    
    % Compute gradients in x and y directions
    gradientX = imfilter(smoothedImage, SobelX, 'same');
    gradientY = imfilter(smoothedImage, SobelY, 'same');
    
    % Compute gradient magnitude and direction (angle)
    gradientMagnitude = sqrt(double(gradientX.^2 + gradientY.^2));
    gradientDirection = atan2(double(gradientY), double(gradientX));  % Radians, direction of edges

    % Initialize an empty image to store non-maxima suppressed edges
    [nRows, nCols] = size(gradientMagnitude);
    nonMaxSuppressed = zeros(nRows, nCols);
    
    for i = 2:nRows-1
        for j = 2:nCols-1
            % Get gradient direction (quantize to nearest 0, 45, 90, or 135 degrees)
            direction = gradientDirection(i, j) * (180 / pi);  % Convert to degrees
            if direction < 0
                direction = direction + 180;  % Normalize to [0, 180] range
            end
            
            % Check pixel against neighbors based on gradient direction
            if ((direction >= 0 && direction < 22.5) || (direction >= 157.5 && direction <= 180))
                neighbors = [gradientMagnitude(i, j-1), gradientMagnitude(i, j+1)];  % Left, right
            elseif (direction >= 22.5 && direction < 67.5)
                neighbors = [gradientMagnitude(i-1, j+1), gradientMagnitude(i+1, j-1)];  % Top-right, bottom-left
            elseif (direction >= 67.5 && direction < 112.5)
                neighbors = [gradientMagnitude(i-1, j), gradientMagnitude(i+1, j)];  % Top, bottom
            else
                neighbors = [gradientMagnitude(i-1, j-1), gradientMagnitude(i+1, j+1)];  % Top-left, bottom-right
            end
            
            % Suppress non-maximum
            if gradientMagnitude(i, j) >= neighbors(1) && gradientMagnitude(i, j) >= neighbors(2)
                nonMaxSuppressed(i, j) = gradientMagnitude(i, j);
            else
                nonMaxSuppressed(i, j) = 0;
            end
        end
    end

    % Define high and low thresholds
    highThreshold = highThresholdValue * max(nonMaxSuppressed(:));
    lowThreshold = lowThresholdValue * max(nonMaxSuppressed(:));
    
    % Initialize the edge map
    edges = zeros(nRows, nCols);
    
    % Strong edges: above high threshold
    strongEdges = nonMaxSuppressed >= highThreshold;
    
    % Weak edges: between low and high thresholds
    weakEdges = (nonMaxSuppressed >= lowThreshold) & (nonMaxSuppressed < highThreshold);
    
    % Track edges by connecting weak edges to strong edges
    for i = 2:nRows-1
        for j = 2:nCols-1
            if strongEdges(i, j)
                edges(i, j) = 1;  % Mark strong edge
            elseif weakEdges(i, j)
                % Check if the weak edge is connected to a strong edge
                if any(any(strongEdges(i-1:i+1, j-1:j+1)))
                    edges(i, j) = 1;  % Promote weak edge to strong edge
                end
            end
        end
    end

    % Show the original grayscale image
    figure;
    subplot(1, 2, 1);  % Create a 1x2 grid of plots, use the first one for the original image
    imshow(grayImage);
    title('Original Grayscale Image');
    
    % Show the edges detected by your Canny edge detector
    subplot(1, 2, 2);  % Use the second plot for the edge-detected image
    imshow(edges);  % 'edges' is the binary edge map from your Canny implementation
    title('Detected Edges (Canny)');

    % Implement the Harris corner dectector
    % Gradient images (from the Sobel operator earlier)
    Ix = gradientX;  % Already computed in the Canny step
    Iy = gradientY;

    % Compute products of gradients
    Ix2 = Ix.^2;
    Iy2 = Iy.^2;
    Ixy = Ix .* Iy;

    % Apply Gaussian filter to smooth the gradient products
    %G = fspecial('gaussian', [5, 5], sigma);

    Ix2 = imfilter(Ix2, G);
    Iy2 = imfilter(Iy2, G);
    Ixy = imfilter(Ixy, G);

    % Harris constant
    k = 0.04;

    % Compute determinant and trace of the structure tensor
    detM = (Ix2 .* Iy2) - (Ixy .^ 2);
    traceM = Ix2 + Iy2;

    % Compute the Harris response
    R = detM - k * (traceM .^ 2);

    % Threshold the response
    cornerThreshold = 0.1 * max(R(:));  % Adjust this as needed
    cornerCandidates = (R > cornerThreshold);

    % Perform non-maximum suppression
    corners = imregionalmax(R) & cornerCandidates;  % Keep only local maxima

    % Plot the detected corners on the original image
    [cornerRow, cornerCol] = find(corners);
    figure;
    imshow(grayImage); hold on;
    plot(cornerCol, cornerRow, 'ro');  % Red dots for corners
    title('Detected Corners (Harris)');


end