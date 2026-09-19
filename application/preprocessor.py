"""Application layer implementing core image preprocessing algorithms."""

from typing import Optional
import cv2
import numpy as np

from core.config import PreprocessorConfig
from core.interfaces import ImagePreprocessorInterface


class FMBImagePreprocessor(ImagePreprocessorInterface):
    """Core image preprocessor for Field Measurement Book (FMB) survey sketches.

    Transforms raw scanned or photographed map images into binarized representations
    with enhanced boundary line continuity and noise suppression.

    Strict Architectural Constraint:
    This class contains ZERO file I/O operations and operates strictly on in-memory
    NumPy arrays.
    """

    def __init__(self, default_config: Optional[PreprocessorConfig] = None) -> None:
        """Initialize preprocessor with an optional default configuration."""
        self._default_config = default_config or PreprocessorConfig()

    def process(
        self, image: np.ndarray, config: Optional[PreprocessorConfig] = None
    ) -> np.ndarray:
        """Execute the preprocessing pipeline on a raw NumPy image array.

        Pipeline Stages:
            1. Input validation & sanity checks.
            2. Grayscale color conversion (BGR/BGRA -> Grayscale).
            3. Gaussian blur filtering (noise reduction).
            4. Otsu's binarization (optimal foreground/background line separation).

        Args:
            image: Input raw image array (2D grayscale or 3D multi-channel).
            config: Optional override configuration. Defaults to instance config.

        Returns:
            np.ndarray: Clean binarized 2D NumPy array with dtype uint8 (values 0 and 255).

        Raises:
            ValueError: If input image is not a valid non-empty NumPy array.
        """
        active_config = config or self._default_config
        self._validate_input(image)

        # Stage 1: Color Space Conversion to 8-bit Grayscale
        gray = self._to_grayscale(image)

        # Stage 2: Gaussian Blur for High-Frequency Noise Reduction
        blurred = cv2.GaussianBlur(
            gray,
            active_config.blur_kernel_size,
            sigmaX=active_config.blur_sigma_x,
            sigmaY=active_config.blur_sigma_y,
        )

        # Stage 3: Otsu's Adaptive Thresholding
        _, binary = cv2.threshold(
            blurred,
            active_config.threshold_value,
            active_config.max_value,
            active_config.threshold_type,
        )

        return binary

    @staticmethod
    def _validate_input(image: np.ndarray) -> None:
        """Validate input image array integrity."""
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Input must be a numpy.ndarray, got {type(image).__name__}")
        if image.size == 0:
            raise ValueError("Input image array is empty (size 0).")
        if image.ndim not in (2, 3):
            raise ValueError(f"Input image must have 2 or 3 dimensions, got {image.ndim}")

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Ensure image is a single-channel 8-bit grayscale array."""
        # Convert floating-point normalized representations [0.0, 1.0] to uint8 [0, 255]
        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                working_image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                working_image = image.clip(0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            working_image = image.astype(np.uint8)
        else:
            working_image = image

        if working_image.ndim == 2:
            return working_image

        channels = working_image.shape[2]
        if channels == 1:
            return working_image[:, :, 0]
        elif channels == 3:
            return cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            return cv2.cvtColor(working_image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported number of image channels: {channels}")

