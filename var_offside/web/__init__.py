"""
Browser-based VAR offside application.

This package wraps the existing :mod:`var_offside` pipeline in a FastAPI
server, a live 3D renderer and a static frontend.  The PyQt desktop app in
``var_offside.app`` is untouched and keeps working independently.
"""

__all__ = ["phases", "sessions", "jobs", "server"]
