function world_points = triangulatePoints(P1, P2, u1, v1, u2, v2)
    % Triangulates 3D points from 2D correspondences in stereo images.
    %
    % Inputs:
    %   P1 - 3x4 projection matrix for the first camera
    %   P2 - 3x4 projection matrix for the second camera
    %   u1, v1 - Vectors of image coordinates from the first camera
    %   u2, v2 - Vectors of image coordinates from the second camera
    %
    % Outputs:
    %   world_points - Nx3 matrix containing the 3D coordinates of the points

    nPoints = length(u1);  % Number of points
    world_points = zeros(nPoints, 3);  % Initialize output matrix

    for i = 1:nPoints
        % Form the homogeneous 2D points
        x1 = [u1(i); v1(i); 1];
        x2 = [u2(i); v2(i); 1];

        % Formulate the DLT system for each point
        % We will have a system of the form A * X = 0, where X is the 3D point

        % First two rows from the first camera
        A(1,:) = x1(1) * P1(3,:) - P1(1,:);
        A(2,:) = x1(2) * P1(3,:) - P1(2,:);

        % First two rows from the second camera
        A(3,:) = x2(1) * P2(3,:) - P2(1,:);
        A(4,:) = x2(2) * P2(3,:) - P2(2,:);

        % Solve for the 3D point X using SVD
        [~, ~, V] = svd(A);
        X = V(:, end);  % The solution is the last column of V

        % Convert the homogeneous point back to 3D
        world_points(i, :) = X(1:3)' / X(4);
    end
end
