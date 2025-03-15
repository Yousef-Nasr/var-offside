import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math
from detect import get_nearest_boxes_to_side, yolo_inference
from vanshing_point import segment_green, detect_lines, calculate_vanishing_point
from utils import visualize_vanishing_point

from pose import get_pose_keypoints

def visualize_offside_detection(image, vanishing_point, boxes, side='left'):
    """
    Visualize the offside detection by drawing the vanishing point, offside line,
    pose keypoints, and highlighting the detected players.
    
    Args:
        image: The original image
        vanishing_point: The calculated vanishing point (x, y)
        boxes: List of bounding boxes for detected players
        side: 'left' or 'right' side for offside line
    """
    # Get pose keypoints for all players
    all_keypoints = get_pose_keypoints(image, boxes)
    if vanishing_point is None:
        print("No vanishing point found")
        return
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Convert to RGB for matplotlib display
    if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    else:
        vis_image_rgb = vis_image
    
    # Create figure for visualization
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(111)
    ax.imshow(vis_image_rgb)
    
    # Extract vanishing point coordinates
    vp_x, vp_y = vanishing_point
    h, w = image.shape[:2]
    
    # Draw the offside line based on the specified side
    if side.lower() == 'left':
        side_point = (0, h//2)
        line_color = 'r-'
    else:  # right side
        side_point = (w-1, h//2)
        line_color = 'r-'
    
    # Draw the offside line
    ax.plot([vp_x, side_point[0]], [vp_y, side_point[1]], line_color, linewidth=3, label='Offside Line')
    
    # Calculate line parameters (ax + by + c = 0)
    a = side_point[1] - vp_y
    b = vp_x - side_point[0]
    c = side_point[0]*vp_y - vp_x*side_point[1]
    
    # Draw bounding boxes for detected players
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # Get player's pose keypoints
        keypoints = all_keypoints[i] if i < len(all_keypoints) and all_keypoints[i] is not None else None
        
        # Find the lowest keypoint (closest to feet) if available
        if keypoints is not None:
            # Get all valid keypoints (confidence > 0)
            valid_keypoints = keypoints[keypoints[:, 2] > 0]
            if len(valid_keypoints) > 0:
                # Find the keypoint with highest y-coordinate (lowest in image)
                lowest_keypoint_idx = np.argmax(valid_keypoints[:, 1])
                bottom_center_x = valid_keypoints[lowest_keypoint_idx, 0]
                bottom_center_y = valid_keypoints[lowest_keypoint_idx, 1]
            else:
                # Fallback to bounding box if no valid keypoints
                bottom_center_x = (x1 + x2) / 2
                bottom_center_y = y2
        else:
            # Fallback to bounding box if no keypoints
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2
            
        # Draw pose keypoints if available
        if keypoints is not None:
            # Draw keypoints
            for kpt in keypoints:
                if kpt[2] > 0:  # Only draw if confidence is > 0
                    ax.scatter(kpt[0], kpt[1], c=box_color, s=20, alpha=0.7)
            
            # Draw skeleton lines between keypoints
            # Define keypoint pairs for YOLO pose skeleton (17 keypoints)
            skeleton = [
                [5,7], [7,9],   # Left arm
                [6,8], [8,10],  # Right arm
                [5,6],          # Shoulders
                [5,11], [6,12], # Torso
                [11,13], [13,15], # Left leg
                [12,14], [14,16], # Right leg
                [11,12],        # Hips
                [0,1], [1,2], [2,3], [3,4] # Face
            ]
            
            # Only draw skeleton if we have valid keypoints
            if keypoints is not None and len(keypoints) > 0:
                for pair in skeleton:
                    if pair[0] < len(keypoints) and pair[1] < len(keypoints) and \
                       keypoints[pair[0], 2] > 0 and keypoints[pair[1], 2] > 0:
                        ax.plot([keypoints[pair[0], 0], keypoints[pair[1], 0]],
                               [keypoints[pair[0], 1], keypoints[pair[1], 1]],
                               c=box_color, linewidth=1, alpha=0.7)
        
        # Calculate the point on the offside line with the same y-coordinate as the player's feet
        # Line equation: ax + by + c = 0
        # Solving for x when y is fixed: x = -(by + c) / a
        if abs(a) < 1e-8:  # If line is horizontal (unlikely for offside line)
            # Use perpendicular distance as fallback
            distance = abs(a*bottom_center_x + b*bottom_center_y + c) / math.sqrt(a**2 + b**2)
        else:
            # Calculate x-coordinate on the line at the same y-coordinate as the player's feet
            line_x_at_player_y = -(b*bottom_center_y + c) / a
            
            # Calculate horizontal distance from player to the line
            if side.lower() == 'left':
                # For left side, player is offside if they're to the right of the line
                distance = max(0, bottom_center_x - line_x_at_player_y)
            else:  # right side
                # For right side, player is offside if they're to the left of the line
                distance = max(0, line_x_at_player_y - bottom_center_x)
        
        # Check if player is on the correct side of the vanishing point
        is_correct_side = False
        if side.lower() == 'left':
            is_correct_side = bottom_center_x < vp_x
        else:  # right side
            is_correct_side = bottom_center_x > vp_x
        
        # Determine box color based on distance to line
        if i == 0 and is_correct_side:  # Highlight the nearest player
            box_color = 'red'
            label = 'Nearest Player'
        elif is_correct_side:
            box_color = 'blue'
            label = f'Player {i+1}'
        else:
            box_color = 'gray'
            label = f'Player {i+1} (wrong side)'
        
        # Draw rectangle and bottom center point
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        ax.scatter(bottom_center_x, bottom_center_y, c=box_color, s=50)
        
        # Calculate the nearest point on the offside line to the player's position
        # Formula for nearest point on line ax + by + c = 0 from point (x0, y0):
        # x = (b*(b*x0 - a*y0) - a*c) / (a*a + b*b)
        # y = (a*(-b*x0 + a*y0) - b*c) / (a*a + b*b)
        denom = a*a + b*b
        nearest_x = (b*(b*bottom_center_x - a*bottom_center_y) - a*c) / denom
        nearest_y = (a*(-b*bottom_center_x + a*bottom_center_y) - b*c) / denom
        
        # Draw a horizontal line from player's position to the nearest point on the offside line (0 degrees)
        # This ensures we're showing the orthogonal distance correctly
        ax.plot([bottom_center_x, nearest_x], [bottom_center_y, nearest_y], 
                color=box_color, linestyle='--', linewidth=1.5)
        
        # Add a small marker at the intersection point
        ax.scatter(nearest_x, nearest_y, c=box_color, s=30, alpha=0.7)
        
        # Add distance label on the line with more visibility
        mid_x = (bottom_center_x + nearest_x) / 2
        mid_y = (bottom_center_y + nearest_y) / 2
        ax.text(mid_x, mid_y, f'{distance:.1f}', color=box_color, fontsize=10, 
                backgroundcolor='white', ha='center', va='center', weight='bold')
        
        # Add label with player info
        if is_correct_side:
            ax.text(x1, y1-10, f'{label} (d={distance:.1f})', color=box_color, fontsize=10, backgroundcolor='white')
        else:
            ax.text(x1, y1-10, label, color=box_color, fontsize=10, backgroundcolor='white')
    
    # Plot the vanishing point
    ax.scatter(vp_x, vp_y, c='red', s=100, label='Vanishing Point')
    
    # Add title and legend
    plt.title(f'Offside Detection ({side.capitalize()} Side)', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Invert y-axis to match image coordinates
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def demo_offside_detection(image_path, side='left', n_players=5):
    """
    Demonstrate the offside detection process on an image.
    
    Args:
        image_path: Path to the input image
        side: 'left' or 'right' side for offside detection
        n_players: Number of nearest players to detect
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Step 1: Segment the green field
    green_only, mask = segment_green(image)
    
    # Step 2: Detect lines on the field
    lines, edges = detect_lines(green_only)
    
    # Step 3: Calculate vanishing point
    vanishing_point, candidate_points = calculate_vanishing_point(lines, image)
    if vanishing_point is None:
        print("Could not find vanishing point")
        return
    
    # Step 4: Detect players using YOLO
    results = yolo_inference(image_path)
    
    # Step 5: Get nearest players to the specified side
    nearest_boxes = get_nearest_boxes_to_side(
        results, 
        side=side, 
        n=n_players, 
        img_width=image.shape[1], 
        vanishing_point=vanishing_point
    )
    
    # Step 6: Visualize the results
    visualize_offside_detection(image, vanishing_point, nearest_boxes, side)
    
    # Also show the vanishing point with field lines
    visualize_vanishing_point(image, lines, vanishing_point, candidate_points, side)

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "test7.png"  # Update with your test image path
    
    # Demonstrate offside detection for left side
    demo_offside_detection(image_path, side='left', n_players=2)
    
    # Optionally demonstrate for right side
    # demo_offside_detection(image_path, side='right', n_players=5)