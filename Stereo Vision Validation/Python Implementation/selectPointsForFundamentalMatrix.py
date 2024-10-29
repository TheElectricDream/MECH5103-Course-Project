import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Disable scientific notation and set precision
np.set_printoptions(suppress=True,  # Suppress scientific notation
                   precision=4,      # Number of decimal places
                   floatmode='fixed')  # Fixed number of decimal places

class PointSelector:
    def __init__(self, image_path1, image_path2, n_points):
        # Read images
        self.img1 = cv2.imread(image_path1)
        self.img2 = cv2.imread(image_path2)
        # Convert from BGR to RGB
        self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        
        self.n_points = n_points
        self.current_point = 0
        self.current_image = 1  # Start with image 1
        
        # Initialize arrays for points
        self.u1 = np.zeros(n_points)
        self.v1 = np.zeros(n_points)
        self.u2 = np.zeros(n_points)
        self.v2 = np.zeros(n_points)
        
        # Setup the plot
        self.setup_plot()
        
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.img_plot = self.ax.imshow(self.img1)
        self.ax.axis('equal')
        
        # Add zoom button
        self.bzoom = Button(plt.axes([0.8, 0.025, 0.1, 0.04]), 'Zoom')
        self.bzoom.on_clicked(self.toggle_zoom)
        
        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.title(f'Select point {self.current_point + 1} of {self.n_points} on Image {self.current_image}')
        
    def toggle_zoom(self, event):
        if self.ax.get_navigate_mode():
            self.ax.set_navigate_mode(None)
            self.bzoom.label.set_text('Zoom')
        else:
            self.ax.set_navigate_mode('zoom')
            self.bzoom.label.set_text('Done')
            
    def on_click(self, event):
        if event.inaxes != self.ax or self.ax.get_navigate_mode():
            return
            
        if self.current_image == 1:
            self.u1[self.current_point] = event.xdata
            self.v1[self.current_point] = event.ydata
            self.ax.scatter(event.xdata, event.ydata, c='r', marker='o')
        else:
            self.u2[self.current_point] = event.xdata
            self.v2[self.current_point] = event.ydata
            self.ax.scatter(event.xdata, event.ydata, c='r', marker='o')
            
        self.fig.canvas.draw()
        
        # Move to next point or switch to second image
        if self.current_image == 1 and self.current_point == self.n_points - 1:
            self.current_image = 2
            self.current_point = 0
            self.img_plot.set_data(self.img2)
            self.ax.clear()
            self.img_plot = self.ax.imshow(self.img2)
            self.ax.axis('equal')
        elif self.current_image == 1:
            self.current_point += 1
        elif self.current_image == 2 and self.current_point < self.n_points - 1:
            self.current_point += 1
        else:
            plt.close()
            return
            
        plt.title(f'Select point {self.current_point + 1} of {self.n_points} on Image {self.current_image}')
        
    def get_points(self):
        return (self.u1, self.v1), (self.u2, self.v2)

def select_image_points(image_path1, image_path2, n_points):
    selector = PointSelector(image_path1, image_path2, n_points)
    plt.show()
    return selector.get_points()

# Main function
if __name__ == "__main__":
    nPointsTotal = 10  # Set the number of points you want to select
    (u1, v1), (u2, v2) = select_image_points('stereo_img1_small.jpeg', 'stereo_img2_small.jpeg', nPointsTotal)
    
    # Print the selected points
    print("Points from Image 1:")
    for i in range(len(u1)):
        print(f"Point {i+1}: ({u1[i]:.2f}, {v1[i]:.2f})")
        
    print("\nPoints from Image 2:")
    for i in range(len(u2)):
        print(f"Point {i+1}: ({u2[i]:.2f}, {v2[i]:.2f})")