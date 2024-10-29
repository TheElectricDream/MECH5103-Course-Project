function plotCameraFrame(R, t, labelText)
    % Function to plot a camera frame based on rotation R and translation t
    
    % Camera reference frame axes (unit vectors along X, Y, and Z)
    origin = t;
    xAxis = R * [1; 0; 0];
    yAxis = R * [0; 1; 0];
    zAxis = R * [0; 0; 1];
    
    % Scale factor for the length of the axes
    scale = 0.1;  % Adjust this to make the axes longer/shorter
    
    % Plot the camera origin as a point
    plot3(t(1), t(2), t(3), 'bo', 'MarkerSize', 10, 'LineWidth', 2);  % Blue circle for the camera center
    
    % Plot the axes of the camera
    quiver3(origin(1), origin(2), origin(3), scale*xAxis(1), scale*xAxis(2), scale*xAxis(3), 'r', 'LineWidth', 2); % X-axis (red)
    quiver3(origin(1), origin(2), origin(3), scale*yAxis(1), scale*yAxis(2), scale*yAxis(3), 'g', 'LineWidth', 2); % Y-axis (green)
    quiver3(origin(1), origin(2), origin(3), scale*zAxis(1), scale*zAxis(2), scale*zAxis(3), 'b', 'LineWidth', 2); % Z-axis (blue)
    
    % Add label for the camera
    text(origin(1), origin(2), origin(3), labelText, 'FontSize', 12, 'Color', 'k');
end