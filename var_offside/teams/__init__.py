"""Teams module - Team assignment using embeddings and clustering."""

from var_offside.teams.embeddings import EmbeddingExtractor
from var_offside.teams.clustering import TeamClusterer
from var_offside.teams.team_assigner import TeamAssigner, TeamAssignmentResult

__all__ = [
    "EmbeddingExtractor",
    "TeamClusterer", 
    "TeamAssigner",
    "TeamAssignmentResult",
]
