"""FLIM Studio: a small, dependency-light FLIM analysis application."""

from .models import AnalysisResult, AnalysisSettings, FLIMDataset
from .processing import analyze_dataset

__all__ = ["AnalysisResult", "AnalysisSettings", "FLIMDataset", "analyze_dataset"]

__version__ = "0.12.10"
