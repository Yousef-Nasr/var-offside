"""
ResNet-based embedding extraction for team clustering.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Optional
import cv2

from var_offside.config import get_settings


class EmbeddingExtractor:
    """
    Extract visual embeddings from player crops using ResNet50.
    These embeddings are used for team clustering.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize embedding extractor.
        
        Args:
            device: Compute device ('cuda' or 'cpu')
        """
        settings = get_settings()
        self.device = device or settings.device
        self.embedding_dim = settings.teams.embedding_dim
        self.jersey_crop_ratio = settings.teams.jersey_crop_ratio
        
        # Initialize model
        self._model: Optional[nn.Module] = None
        
        # Image preprocessing for ResNet
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    @property
    def model(self) -> nn.Module:
        """Lazy load ResNet model."""
        if self._model is None:
            # Load pretrained ResNet50
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            # Remove classification layer to get embeddings
            self._model = nn.Sequential(*list(resnet.children())[:-1])
            self._model = self._model.to(self.device)
            self._model.eval()
        
        return self._model
    
    def extract_crop(
        self,
        image: np.ndarray,
        bbox: tuple,
        use_jersey_area: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract crop from image for embedding.
        
        Args:
            image: Full BGR image
            bbox: (x1, y1, x2, y2) bounding box
            use_jersey_area: Focus on upper body (jersey)
            
        Returns:
            RGB crop ready for embedding, or None if invalid
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = image.shape[:2]
        
        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Get crop
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None
        
        # Focus on jersey area (upper body)
        if use_jersey_area and crop.shape[0] > 40:
            jersey_height = int(crop.shape[0] * self.jersey_crop_ratio)
            crop = crop[:jersey_height, :]
        
        # Convert BGR to RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        return crop
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        crops: List[np.ndarray]
    ) -> np.ndarray:
        """
        Extract embeddings from a list of crops.
        
        Args:
            crops: List of RGB image crops
            
        Returns:
            Array of embeddings (N, embedding_dim)
        """
        if not crops:
            return np.array([])
        
        # Preprocess all crops
        tensors = []
        for crop in crops:
            try:
                tensor = self.preprocess(crop)
                tensors.append(tensor)
            except Exception:
                # Skip invalid crops
                continue
        
        if not tensors:
            return np.array([])
        
        # Stack into batch
        batch = torch.stack(tensors).to(self.device)
        
        # Extract embeddings
        embeddings = self.model(batch)
        embeddings = embeddings.squeeze(-1).squeeze(-1)  # Remove spatial dims
        
        return embeddings.cpu().numpy()
    
    def extract_from_detections(
        self,
        image: np.ndarray,
        bboxes: List[tuple],
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Extract embeddings for all detected players.
        
        Args:
            image: Full BGR image
            bboxes: List of (x1, y1, x2, y2) bounding boxes
            batch_size: Batch size for inference
            
        Returns:
            Embeddings array (N, embedding_dim)
        """
        # Extract crops
        crops = []
        valid_indices = []
        
        for i, bbox in enumerate(bboxes):
            crop = self.extract_crop(image, bbox)
            if crop is not None:
                crops.append(crop)
                valid_indices.append(i)
        
        if not crops:
            return np.zeros((len(bboxes), self.embedding_dim))
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_embeddings = self.extract_embeddings(batch_crops)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Create full result array with zeros for invalid crops
        result = np.zeros((len(bboxes), self.embedding_dim))
        for i, valid_idx in enumerate(valid_indices):
            if i < len(embeddings):
                result[valid_idx] = embeddings[i]
        
        return result
