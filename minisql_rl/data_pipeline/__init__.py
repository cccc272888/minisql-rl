"""Deterministic Text-to-SQL training-data generation."""

from .generate import PipelineConfig, build_training_data

__all__ = ["PipelineConfig", "build_training_data"]
