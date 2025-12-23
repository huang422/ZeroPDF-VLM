"""Template management module for golden templates and ROI configurations."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2


@dataclass
class ROI:
    """Region of Interest definition from template configuration."""
    roi_id: str
    description: str
    x1: int
    y1: int
    x2: int
    y2: int
    format: str = "top_left_bottom_right"

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    def is_valid(self, image_shape: Tuple[int, int]) -> bool:
        """Check if ROI coordinates are within image bounds."""
        height, width = image_shape[:2]
        return (0 <= self.x1 < self.x2 <= width and
                0 <= self.y1 < self.y2 <= height)


@dataclass
class GoldenTemplate:
    """Golden template for document classification and alignment."""
    template_id: str
    template_image_path: str
    config_path: str
    features_cache_path: str
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None
    image_shape: Optional[Tuple[int, int, int]] = None
    rois: List[ROI] = None

    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.rois is None:
            self.rois = []

    def has_features(self) -> bool:
        """Check if SIFT features are loaded."""
        return self.keypoints is not None and self.descriptors is not None

    def feature_count(self) -> int:
        """Return number of SIFT features."""
        return len(self.keypoints) if self.keypoints else 0


__all__ = ['ROI', 'GoldenTemplate']
