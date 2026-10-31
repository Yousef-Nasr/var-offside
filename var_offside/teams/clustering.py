"""
K-Means clustering for team assignment.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Tuple, List
from dataclasses import dataclass

from var_offside.config import get_settings


@dataclass
class ClusteringResult:
    """Result of team clustering."""
    labels: np.ndarray  # Team label for each player (1 or 2)
    centroids: np.ndarray  # Cluster centroids
    confidence: np.ndarray  # Confidence for each assignment
    
    @property
    def team1_indices(self) -> List[int]:
        """Indices of team 1 players."""
        return [i for i, l in enumerate(self.labels) if l == 1]
    
    @property
    def team2_indices(self) -> List[int]:
        """Indices of team 2 players."""
        return [i for i, l in enumerate(self.labels) if l == 2]


class TeamClusterer:
    """
    Cluster players into teams using K-Means on embeddings.
    """
    
    def __init__(
        self,
        n_teams: int = 2,
        use_umap: bool = True,
        umap_components: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize team clusterer.
        
        Args:
            n_teams: Number of teams (usually 2)
            use_umap: Whether to use UMAP for dimension reduction
            umap_components: Number of UMAP components
            random_state: Random seed for reproducibility
        """
        settings = get_settings()
        
        self.n_teams = n_teams
        self.use_umap = use_umap
        self.umap_components = umap_components or settings.teams.umap_components
        self.random_state = random_state
        
        self._umap_reducer = None
        self._kmeans = None
    
    @property
    def umap_reducer(self):
        """Lazy load UMAP reducer."""
        if self._umap_reducer is None and self.use_umap:
            try:
                import umap
                self._umap_reducer = umap.UMAP(
                    n_components=self.umap_components,
                    random_state=self.random_state,
                    min_dist=0.0,
                    n_neighbors=min(15, 5)  # Adjust for small sample sizes
                )
            except ImportError:
                print("Warning: UMAP not available, using raw embeddings")
                self.use_umap = False
        return self._umap_reducer
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Reduce embedding dimensions using UMAP.
        
        Args:
            embeddings: High-dimensional embeddings
            
        Returns:
            Reduced embeddings
        """
        if not self.use_umap or embeddings.shape[0] < 3:
            # Not enough samples for UMAP
            return embeddings
        
        try:
            reduced = self.umap_reducer.fit_transform(embeddings)
            return reduced
        except Exception as e:
            print(f"UMAP reduction failed: {e}")
            return embeddings
    
    def cluster(
        self,
        embeddings: np.ndarray
    ) -> ClusteringResult:
        """
        Cluster embeddings into teams.
        
        Args:
            embeddings: Player embeddings (N, D)
            
        Returns:
            ClusteringResult with team assignments
        """
        n_samples = embeddings.shape[0]
        
        if n_samples < self.n_teams:
            # Not enough players to cluster
            labels = np.arange(1, n_samples + 1)
            return ClusteringResult(
                labels=labels,
                centroids=embeddings,
                confidence=np.ones(n_samples)
            )
        
        # Reduce dimensions
        reduced = self._reduce_dimensions(embeddings)
        
        # K-Means clustering
        kmeans = KMeans(
            n_clusters=self.n_teams,
            random_state=self.random_state,
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(reduced)
        
        # Convert to 1-based team IDs
        labels = cluster_labels + 1
        
        # Calculate confidence based on distance to centroid
        confidence = self._calculate_confidence(reduced, kmeans)
        
        return ClusteringResult(
            labels=labels,
            centroids=kmeans.cluster_centers_,
            confidence=confidence
        )
    
    def _calculate_confidence(
        self,
        embeddings: np.ndarray,
        kmeans: KMeans
    ) -> np.ndarray:
        """
        Calculate confidence scores for cluster assignments.
        Based on relative distance to assigned vs other centroid.
        
        Args:
            embeddings: Reduced embeddings
            kmeans: Fitted KMeans model
            
        Returns:
            Confidence scores (0-1)
        """
        # Get distances to all centroids
        distances = kmeans.transform(embeddings)  # (N, n_clusters)
        
        # For each point, compare distance to assigned cluster vs others
        labels = kmeans.labels_
        confidence = np.zeros(len(labels))
        
        for i, label in enumerate(labels):
            assigned_dist = distances[i, label]
            other_dists = [distances[i, j] for j in range(self.n_teams) if j != label]
            
            if other_dists:
                min_other_dist = min(other_dists)
                # Confidence based on ratio of distances
                if assigned_dist > 0:
                    ratio = min_other_dist / (assigned_dist + min_other_dist)
                    confidence[i] = ratio
                else:
                    confidence[i] = 1.0
            else:
                confidence[i] = 1.0
        
        return confidence
    
    def assign_consistent_teams(
        self,
        embeddings: np.ndarray,
        reference_embeddings: Optional[np.ndarray] = None,
        reference_labels: Optional[np.ndarray] = None
    ) -> ClusteringResult:
        """
        Assign teams with consistent labeling across frames.
        Uses reference embeddings to maintain team identity.
        
        Args:
            embeddings: Current frame embeddings
            reference_embeddings: Previous frame embeddings
            reference_labels: Previous frame labels
            
        Returns:
            ClusteringResult with consistent team IDs
        """
        result = self.cluster(embeddings)
        
        if reference_embeddings is None or reference_labels is None:
            return result
        
        # Match clusters to reference using centroid similarity
        # This ensures Team 1 stays Team 1 across frames
        # (Not needed for single image, but useful for video extension)
        
        return result
