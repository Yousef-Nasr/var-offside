import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.cluster import KMeans
import supervision as sv


def visualize_lines(image, filtered_lines, edges):
    # Create copies for visualization
    line_image = image.copy()
    edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Draw detected lines
    if filtered_lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display results
    plt.figure(figsize=(30, 10))
    plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(132), plt.imshow(cv2.cvtColor(edge_image, cv2.COLOR_BGR2RGB))
    plt.title('Edge Detection'), plt.axis('off')
    plt.subplot(133), plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Lines'), plt.axis('off')
    plt.show()
    return line_image

def visualize_vanishing_point(image, lines, vanishing_point, candidate_points=None, side=None):
    """
    Visualize the vanishing point and lines extending from the image to the vanishing point.
    
    Args:
        image: The original image
        lines: The detected lines in the image
        vanishing_point: The calculated vanishing point [x, y]
        candidate_points: Optional list of candidate vanishing points for visualization
    """
    if vanishing_point is None:
        print("No vanishing point found")
        return
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Convert to RGB for matplotlib display if needed
    if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    else:
        vis_image_rgb = vis_image
    
    # Draw the original lines
    for line in lines:
        if len(line) == 1 and len(line[0]) == 4:  # HoughLinesP format
            x1, y1, x2, y2 = line[0]
            cv2.line(vis_image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        elif len(line) == 2:  # HoughLines format
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(vis_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Create figure for visualization with adjusted figure size
    plt.figure(figsize=(15, 15))
    
    # Create subplot with adjusted aspect ratio
    ax = plt.subplot(111)
    ax.imshow(vis_image_rgb)
    
    # Draw lines from image edges to vanishing point
    vp_x, vp_y = vanishing_point
    h, w = image.shape[:2]
    
    # Define key points on the image edges
    edge_points = [
        (0, 0),          # Top-left
        (w-1, 0),        # Top-right
        (0, h-1),        # Bottom-left
        (w-1, h-1),      # Bottom-right
        (w//2, 0),       # Top-middle
        (w//2, h-1),     # Bottom-middle
        (0, h//2),       # Left-middle
        (w-1, h//2)      # Right-middle
    ]
    
    # Draw lines from edge points to vanishing point
    for pt in edge_points:
        ax.plot([pt[0], vp_x], [pt[1], vp_y], 'b-', alpha=0.5)
    
    # Plot candidate points if provided
    if candidate_points is not None and len(candidate_points) > 0:
        candidate_points = np.array(candidate_points)
        ax.scatter(candidate_points[:, 0], candidate_points[:, 1], c='blue', alpha=0.2, s=10, label='Candidate Points')
    
    # Plot the vanishing point
    ax.scatter(vp_x, vp_y, c='red', s=100, label='Vanishing Point')
    
    # Add labels and title
    plt.title('Vanishing Point Visualization', fontsize=16)
    
    # Set axis limits to match the image dimensions
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Invert y-axis to match image coordinates
    
    # Remove axis ticks for cleaner visualization
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend with better positioning
    plt.legend(loc='upper right', fontsize=12)
    
    # Adjust layout to use the full figure space
    plt.tight_layout()
    plt.show()