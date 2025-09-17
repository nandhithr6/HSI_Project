"""
Package Initializer for the Spatial Stream Module.

This file makes the core components of the spatial stream pipeline—the local
stream, the global stream, and the fusion module—directly importable from the
`spatial_stream` package.

This simplifies imports in higher-level modules, such as the main model file.

Author: Nandhitha
Date: September 17, 2025
Version: 1.0.0
"""

# Import the main class from each component file
from .local_main import LocalFeatureStream
from .spatial_tokenizer import SpatialTokenizer

# Optionally keep other imports if needed
from .global_main import GlobalFeatureStream
from .fusion_main import SpatialFusionModule

# Define the public API of this package.
__all__ = [
    "LocalFeatureStream",
    "SpatialTokenizer",
    "GlobalFeatureStream",
    "SpatialFusionModule"
]
# When a user writes `from models.spatial_stream import *`, only these
# names will be imported. This is a best practice for clean code.
__all__ = [
    "LocalFeatureStream",
    "GlobalFeatureStream",
    "SpatialFusionModule"
]

