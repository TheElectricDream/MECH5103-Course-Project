function X = triangulatePoint(u1, v1, u2, v2, R, t)
    % Triangulate 3D point from two image coordinates (u1, v1) and (u2, v2)
    % Inputs:
    % - u1, v1: 2D image coordinates in the first image
    % - u2, v2: 2D image coordinates in the second image
    % - R, t: Rotation and translation between the two cameras
    % Output:
    % - X: 3D coordinates of the triangulated point

    % Define the projection matrices for both cameras
    P1 = [eye(3), zeros(3,1)];  % First camera (Identity rotation, no translation)
    P2 = [R, t];                % Second camera (Rotation R, translation t)
    
    % Set up the linear system of equations
    A = [
        u1 * P1(3,:) - P1(1,:);
        v1 * P1(3,:) - P1(2,:);
        u2 * P2(3,:) - P2(1,:);
        v2 * P2(3,:) - P2(2,:);
    ];
    
    % Solve using SVD (Singular Value Decomposition)
    [~, ~, V] = svd(A);
    X_homogeneous = V(:, end);
    
    % Convert homogeneous coordinates to 3D coordinates (normalize by the last element)
    X = X_homogeneous(1:3) / X_homogeneous(4);
end