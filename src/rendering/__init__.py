"""
Rendering and visual annotation package for CCTV pipeline.
"""
from src.rendering.annotator import (
    Annotator,
    CyberpunkHUDAnnotator,
    FrameAnnotator,
    NoOpAnnotator,
    ScriptedAnnotator
)

__all__ = [
    "Annotator",
    "CyberpunkHUDAnnotator",
    "FrameAnnotator",
    "NoOpAnnotator",
    "ScriptedAnnotator"
]
