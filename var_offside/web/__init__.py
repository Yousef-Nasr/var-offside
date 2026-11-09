"""
Browser-based VAR offside application.

This package wraps the existing :mod:`var_offside` pipeline in a FastAPI
server, a live 3D renderer and a static frontend.  Start it with
``var-offside-web`` or ``python -m var_offside.web``.
"""

__all__ = ["phases", "sessions", "jobs", "server"]
