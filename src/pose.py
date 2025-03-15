from ultralytics import YOLO
import numpy as np

model_pose = YOLO('yolo11x-pose.pt')

def get_pose_keypoints(image, boxes):
    """Get pose keypoints for specific players based on their bounding boxes.
    
    Args:
        image: The input image
        boxes: List of bounding boxes for players to analyze
        
    Returns:
        List of keypoints for each player
    """
    results = model_pose(image, conf=0.5)
    all_keypoints = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Crop the image to the player's bounding box
        player_img = image[y1:y2, x1:x2]
        if player_img.size == 0:
            continue
            
        # Get pose for the cropped player image
        player_pose = model_pose(player_img, conf=0.5)
        if len(player_pose) > 0 and hasattr(player_pose[0], 'keypoints') and len(player_pose[0].keypoints) > 0:
            # Get keypoints and adjust coordinates to original image
            kpts = player_pose[0].keypoints.data[0].cpu().numpy()
            # Adjust keypoint coordinates back to original image space
            kpts[:, 0] += x1
            kpts[:, 1] += y1
            all_keypoints.append(kpts)
        else:
            all_keypoints.append(None)
    
    return all_keypoints