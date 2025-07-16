"""
05_pipeline - Complete Pipeline Assembly

This module puts everything together into easy-to-use pipelines.
Perfect for when you understand the components and want to run
the complete workflow.

🎯 Learning Path:
1. Start with pipeline_runner.py - simplest complete workflow
2. Explore preprocessing_pipeline.py - full control over the process
3. Try config_manager.py - customize behavior for your needs
4. Use in production with full pipeline

🚀 What this provides:
- SimpleRunner: One-line preprocessing for beginners
- PreprocessingPipeline: Full control for advanced users
- ConfigManager: Easy customization of behavior
- Complete workflows from raw data to SVM-ready features

🧠 Key Benefits:
- Hides complexity while preserving power
- Handles all the orchestration for you
- Provides sensible defaults but allows customization
- Includes error handling and progress reporting

Components:
- PipelineRunner: Simple, one-line interfaces
- PreprocessingPipeline: Full-featured pipeline with all options
- ConfigManager: Configuration and customization tools
"""

from .pipeline_runner import SimpleRunner, AdvancedRunner
from .preprocessing_pipeline import PreprocessingPipeline
from .config_manager import ConfigManager

__all__ = [
    'SimpleRunner',
    'AdvancedRunner', 
    'PreprocessingPipeline',
    'ConfigManager'
]