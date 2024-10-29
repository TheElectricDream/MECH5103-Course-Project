function X = triangulate_points(P1, P2, x1, x2)
    % P1, P2: 3x4 camera projection matrices for the two cameras
    % x1, x2: 2xN matrices containing the 2D points in each image (homogeneous coordinates)
    % X: 4xN matrix of the triangulated 3D points (in homogeneous coordinates)
    
    num_points = size(x1, 2);  % Number of points
    X = zeros(4, num_points);  % Output matrix for 3D points
    
    for i = 1:num_points
        % Extract corresponding points in both images
        x1_i = x1(:, i);
        x2_i = x2(:, i);
        
        % Set up the linear system (Ax = 0)
        A = [
            x1_i(1) * P1(3,:) - P1(1,:);
            x1_i(2) * P1(3,:) - P1(2,:);
            x2_i(1) * P2(3,:) - P2(1,:);
            x2_i(2) * P2(3,:) - P2(2,:);
        ];
        
        % Solve the linear system using SVD
        [~, ~, V] = svd(A);
        X(:, i) = V(:, end);  % Solution is the last column of V
    end
    
    % Convert from homogeneous to 3D coordinates (X, Y, Z)
    X = X ./ X(4, :);  % Normalize to make homogeneous coordinate 1
    
end
