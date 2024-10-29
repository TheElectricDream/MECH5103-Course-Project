function [points_normalized, T] = normalize_points(points)
    % points: 2xN matrix of 2D points (homogeneous coordinates)
    % points_normalized: 2xN matrix of normalized points
    % T: 3x3 normalization matrix

    % Compute centroid of the points
    centroid = mean(points(1:2,:), 2);

    % Shift the origin to the centroid
    shifted_points = points(1:2,:) - centroid;

    % Compute the average distance from the origin
    avg_dist = mean(sqrt(sum(shifted_points.^2, 1)));

    % Compute the scaling factor
    scale = sqrt(2) / avg_dist;

    % Create the normalization transformation matrix
    T = [scale, 0, -scale * centroid(1);
         0, scale, -scale * centroid(2);
         0, 0, 1];

    % Apply the transformation
    points_normalized = T * points;
end
