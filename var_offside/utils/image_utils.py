"""
Image processing utility functions.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        path: Path to image file
        
    Returns:
        BGR image as numpy array
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image can't be loaded
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    
    return image


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save an image to disk.
    
    Args:
        image: BGR image as numpy array
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_size: Optional[int] = None
) -> np.ndarray:
    """
    Resize image maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
        max_size: Maximum dimension (optional)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max_size is not None:
        if max(h, w) > max_size:
            if h > w:
                height = max_size
                width = int(w * max_size / h)
            else:
                width = max_size
                height = int(h * max_size / w)
    elif width is not None and height is None:
        height = int(h * width / w)
    elif height is not None and width is None:
        width = int(w * height / h)
    elif width is None and height is None:
        return image
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def crop_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop image using bounding box.
    
    Args:
        image: Input image
        bbox: (x1, y1, x2, y2) bounding box
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]
    h, w = image.shape[:2]
    
    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return image[y1:y2, x1:x2]


def segment_by_color(
    image: np.ndarray,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment image by HSV color range.
    
    Args:
        image: BGR input image
        lower_hsv: Lower HSV bounds
        upper_hsv: Upper HSV bounds
        
    Returns:
        Tuple of (segmented_image, mask)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented, mask


def get_jersey_crop(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    ratio: float = 0.5
) -> np.ndarray:
    """
    Extract jersey area from player bounding box.
    Focus on upper body where jersey is visible.
    
    Args:
        image: Full image
        bbox: Player bounding box (x1, y1, x2, y2)
        ratio: Portion of height to use (from top)
        
    Returns:
        Jersey crop
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]
    height = y2 - y1
    
    # Get upper portion (jersey area)
    jersey_y2 = y1 + int(height * ratio)
    
    return crop_image(image, (x1, y1, x2, jersey_y2))
