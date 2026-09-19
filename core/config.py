"""Domain configuration for image preprocessing."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PreprocessorConfig:
    """Configuration settings for the FMB image preprocessing pipeline.

    Attributes:
        blur_kernel_size: Tuple (width, height) specifying Gaussian blur kernel dimensions.
                          Both values must be positive, odd integers.
        blur_sigma_x: Gaussian kernel standard deviation in the X direction.
        blur_sigma_y: Gaussian kernel standard deviation in the Y direction (0 defaults to sigma_x).
        threshold_type: OpenCV thresholding flag. Default is 8 (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU = 1 + 8 = 9,
                        or cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU).
        threshold_value: Base threshold value (used by fixed thresholding or Otsu seed).
        max_value: Non-zero value assigned to pixels for which the condition is satisfied.
        invert_output: If True, foreground lines are white (255) on black (0); if False, standard.
    """

    blur_kernel_size: Tuple[int, int] = (5, 5)
    blur_sigma_x: float = 0.0
    blur_sigma_y: float = 0.0
    # In OpenCV: THRESH_BINARY = 0, THRESH_BINARY_INV = 1, THRESH_OTSU = 8
    # Default for FMB survey maps (dark lines on light paper): 1 | 8 = 9 (THRESH_BINARY_INV + THRESH_OTSU)
    threshold_type: int = 9
    threshold_value: float = 0.0
    max_value: float = 255.0
    invert_output: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        kw, kh = self.blur_kernel_size
        if kw <= 0 or kh <= 0:
            raise ValueError(f"Kernel size dimensions must be positive integers, got: {self.blur_kernel_size}")
        if kw % 2 == 0 or kh % 2 == 0:
            raise ValueError(f"Kernel size dimensions must be odd integers, got: {self.blur_kernel_size}")
        if not (0.0 <= self.max_value <= 255.0):
            raise ValueError(f"max_value must be within [0.0, 255.0], got: {self.max_value}")
        if self.threshold_value < 0.0:
            raise ValueError(f"threshold_value must be non-negative, got: {self.threshold_value}")

