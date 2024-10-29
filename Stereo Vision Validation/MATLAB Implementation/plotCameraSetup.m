function plotCameraSetup(K, R, t)
    % Plot two cameras with their reference frames in 3D space
    figure;
    hold on;
    axis equal;
    grid on;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    
    % Plot the first camera (at the origin, with identity rotation)
    plotCameraFrame(eye(3), [0; 0; 0], 'First Camera');
    
    % Plot the second camera (at position t, with rotation R)
    plotCameraFrame(R, t, 'Second Camera');
    
    % Set plot limits
    % xlim([-1 1] * max(abs(t(1))) * 2);
    % ylim([-1 1] * max(abs(t(2))) * 2);
    % zlim([-1 1] * max(abs(t(3))) * 2);
    
    title('Camera Setup in 3D Space');
    legend('First Camera', 'Second Camera');
end