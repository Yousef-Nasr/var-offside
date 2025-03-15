import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO, SAM
from sklearn.cluster import KMeans
import supervision as sv
import math
from utils import visualize_lines, visualize_vanishing_point

def segment_green(image):
    # Convert from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define optimized range for green color with wider range for better field detection
    lower_green = np.array([35, 45, 45])  # Lower saturation and value thresholds
    upper_green = np.array([85, 255, 255])
    
    # Create initial mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a new mask for filtered contours
    filtered_mask = np.zeros_like(mask)
    
    if len(contours) > 0:
        # Find the largest contour (assuming it's the playing field)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw only the largest contour
        cv2.drawContours(filtered_mask, [largest_contour], -1, 255, -1)
    
    # Apply the filtered mask to the original image
    green_only = cv2.bitwise_and(image, image, mask=filtered_mask)
    
    return green_only, filtered_mask

def filter_lines_by_angle(lines, min_angle=18, max_angle=89):
    filtered_lines = []
    if lines is None:
        return filtered_lines
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:  # Vertical line
            angle = 90.0
        else:
            angle = math.degrees(math.atan2(abs(dy), abs(dx)))
            
        # Keep lines within the specified angle range (18° to 89°)
        if min_angle <= angle <= max_angle:
            filtered_lines.append(line)
            
    return filtered_lines

def detect_lines(segmented_image):
    # Extract saturation channel from HSV for better green patch differentiation
    hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # Apply Gaussian blur to reduce noise while preserving edges
    # blurred = cv2.GaussianBlur(saturation, (5,5), 0)
    
    # Apply Canny edge detection with fixed thresholds
    edges = cv2.Canny(saturation, 30, 50)
    
    # Apply morphological operations to enhance edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Use probabilistic Hough transform with fixed parameters
    height = edges.shape[0]
    min_line_length = height * 0.15
    max_line_gap = height * 0.01
    print(min_line_length, max_line_gap)
    lines = cv2.HoughLinesP(edges, 
                           rho=1.0,           # Reduced for higher precision
                           theta=np.pi/180,    # Reduced angle step for finer detection
                           threshold=50,       # Lower threshold to detect more lines
                           minLineLength=100,  # Reduced to detect shorter lines
                           maxLineGap=10)      # Increased to bridge small gaps
    
    # Filter lines by angle with fixed sensitivity
    filtered_lines = filter_lines_by_angle(lines)
    
    return filtered_lines, edges


def line_intersection(line1, line2):
    """
    Calculate intersection point of two infinite lines (in homogeneous coordinates).
    Returns (x, y) or None if lines are parallel/coincident.
    """
    # Extract line endpoints
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # Convert lines to homogeneous coordinates
    p1 = np.array([x1, y1, 1])
    p2 = np.array([x2, y2, 1])
    p3 = np.array([x3, y3, 1])
    p4 = np.array([x4, y4, 1])

    # Get line equations using cross product
    l1 = np.cross(p1, p2)  # [a, b, c] coefficients for line1
    l2 = np.cross(p3, p4)  # [a, b, c] coefficients for line2

    # Find intersection using cross product of line coefficients
    intersection = np.cross(l1, l2)
    
    # Check if lines are parallel (z-component is 0)
    if abs(intersection[2]) < 1e-8:
        return None
        
    # Convert back to Cartesian coordinates
    x = intersection[0] / intersection[2]
    y = intersection[1] / intersection[2]
    
    return (x, y)

def count_inliers(lines, candidate_point, threshold=5):
    """
    Count how many lines pass close to a candidate vanishing point.
    threshold: Maximum distance in pixels to consider a line an inlier
    """
    inliers = 0
    px, py = candidate_point
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate distance from point to line using line equation
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        
        distance = abs(a*px + b*py + c) / np.sqrt(a**2 + b**2 + 1e-6)
        
        if distance < threshold:
            inliers += 1
            
    return inliers

def calculate_vanishing_point(lines, image):
    if len(lines) < 2:
        return None, []

    # RANSAC algorithm with adaptive parameters
    best_point = None
    max_inliers = 0
    candidate_points = []
    h, w = image.shape[:2]
    
    # Adaptive threshold based on image size
    distance_threshold = min(w, h) * 0.005  # Adaptive threshold for inlier counting
    min_inliers_ratio = 0.3  # Minimum ratio of lines that should support the vanishing point
    
    # Increase iterations for more reliable results
    num_iterations = 500
    
    for _ in range(num_iterations):
        # Select two random lines
        i, j = np.random.randint(0, len(lines), 2)
        line1, line2 = lines[i], lines[j]
        
        # Find intersection
        intersection = line_intersection(line1, line2)
        if intersection is None:
            continue
            
        # Basic outlier rejection - reject points too far outside the image
        margin = 0.5  # Allow points up to 50% outside the image
        if not (-w*margin <= intersection[0] <= w*(1+margin) and
                -h*margin <= intersection[1] <= h*(1+margin)):
            continue
            
        # Store valid candidate point
        candidate_points.append(intersection)
            
        # Count inliers with adaptive threshold
        inliers = count_inliers(lines, intersection, threshold=distance_threshold)
        
        # Update best point if it has more inliers and meets minimum ratio
        if inliers > max_inliers and inliers/len(lines) >= min_inliers_ratio:
            max_inliers = inliers
            best_point = intersection
    
    # Visualize the results
    if best_point is not None:
        # Overlay the vanishing point on the original image
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
        
        # Define key points on the image edges
        h, w = image.shape[:2]
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
            plt.plot([pt[0], best_point[0]], [pt[1], best_point[1]], 'b-', alpha=0.5)
            
        plt.scatter(best_point[0], best_point[1], c='red', s=100, label='Vanishing Point')
        plt.legend()
        plt.title('Vanishing Point on Image')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.show()
    return best_point, candidate_points

from detect import get_nearest_boxes_to_side, yolo_inference
def draw_boxes(boxes, img):
    """
    Draw bounding boxes on an image
    Args:
        boxes: List of lists containing [x1,y1,x2,y2] coordinates
        img: Input image to draw on
    Returns:
        Image with drawn boxes
    """
    # Create a copy of the image to avoid modifying original
    img_copy = img.copy()
    
    # Draw each box
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        # Draw rectangle with green color and 2 pixel thickness
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    return img_copy

if __name__ == "__main__":
    img_path = "test8.png"
    image = cv2.imread(img_path)
    segment_image, mask = segment_green(image)
    sv.plot_image(segment_image)
    filtered_lines, edges = detect_lines(segment_image)
    visualize_lines(image, filtered_lines, edges)
    best_point, candidate_points = calculate_vanishing_point(filtered_lines, image)
    visualize_vanishing_point(image, filtered_lines, best_point, candidate_points)
    result = yolo_inference(img_path)
    boxes = get_nearest_boxes_to_side(result, "right", n=3, vanishing_point=best_point)
    sv.plot_image(draw_boxes(boxes, image))

