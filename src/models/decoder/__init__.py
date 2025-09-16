"""
MSTD+HSH Decoder Package
Author: Abiram
Date: September 16, 2025
"""

from .main import (
    MSTDHSHDecoder,
    TokenToFeatureConverter,
    TransformerBlock,
    CrossScaleAttention,
    MultiScaleFusion,
    HierarchicalSegmentationHead,
    get_device,
    print_memory_usage
)

__all__ = [
    'MSTDHSHDecoder',
    'TokenToFeatureConverter', 
    'TransformerBlock',
    'CrossScaleAttention',
    'MultiScaleFusion',
    'HierarchicalSegmentationHead',
    'get_device',
    'print_memory_usage'
]

__version__ = "1.0.0"
__author__ = "Abiram"
__description__ = "Multi-Scale Transformer Decoder with Hierarchical Segmentation Head for Medical Hyperspectral Image Analysis"
