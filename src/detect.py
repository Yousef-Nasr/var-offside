import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.cluster import KMeans
import supervision as sv



model_yolo = YOLO("best_tuned.pt")

def yolo_inference(img_path):
    yolo_results = model_yolo.predict(img_path, classes=[2])
    return yolo_results[0]


def get_nearest_boxes_to_side(results, side='left', n=3, img_width=None, vanishing_point=None):
    """
    Get the nearest n bounding boxes to a specified side (left or right) from YOLO results.
    
    Args:
        results: YOLO detection results
        side: 'left' or 'right' side of the image
        n: number of boxes to return
        img_width: width of the image (if None, will try to get from results)
        
    Returns:
        List of n bounding boxes nearest to the specified side
    """
    # Extract all bounding boxes from results
    all_boxes = []
    
    # Handle different result formats from YOLO
    if hasattr(results, 'boxes'):
        # For newer YOLO versions
        boxes = results.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            # all_boxes.append({
            #     'x1': x1,
            #     'y1': y1,
            #     'x2': x2,
            #     'y2': y2,
            #     'center_x': (x1 + x2) / 2
            # })
            all_boxes.append([x1, y1, x2, y2])
    elif isinstance(results, list):
        # For results as a list of detections
        for det in results:
            if hasattr(det, 'xyxy'):
                for box in det.xyxy:
                    x1, y1, x2, y2 = box[:4]
                    # all_boxes.append({
                    #     'x1': x1,
                    #     'y1': y1,
                    #     'x2': x2,
                    #     'y2': y2,
                    #     'center_x': (x1 + x2) / 2
                    # })
                    all_boxes.append([x1, y1, x2, y2])
    
    # If no boxes found, return empty list
    if not all_boxes:
        return []
    
    # Determine image width if not provided
    if img_width is None:
        if hasattr(results, 'orig_img'):
            img_width = results.orig_img.shape[1]
        else:
            # Default to maximum x2 value if can't determine width
            img_width = max(box['x2'] for box in all_boxes)
    
    # Sort boxes based on distance to the specified side
    if side.lower() == 'left':
        # Sort by x1 (left edge) for left side
        # sorted_boxes = sorted(all_boxes, key=lambda box: box['x1'])
        sorted_boxes = sorted(all_boxes, key=lambda box: box[0])
    else:
        # Sort by distance from right edge for right side
        # sorted_boxes = sorted(all_boxes, key=lambda box: img_width - box['x2'])
        sorted_boxes = sorted(all_boxes, key=lambda box: img_width - box[2])
    
    # Return the nearest n boxes
    return sorted_boxes[:min(n, len(sorted_boxes))]