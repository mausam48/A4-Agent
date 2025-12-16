"""
Dataset readers for affordance segmentation tasks.

Available datasets:
- UMD Part-Affordance Dataset
- 3DOI Dataset and 3DOI Reasoning Dataset (easy)
"""

from .umd_reader import UmdDataset
from .threedoi_reader import (
    ThreeDOIReasoningDataset,
)

__all__ = ['UmdDataset', 'ThreeDOIReasoningDataset']
