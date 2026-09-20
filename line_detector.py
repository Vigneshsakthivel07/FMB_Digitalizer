"""Module 2: Cadastral Line Detector (line_detector.py).

Detects discrete linear segments and cadastral boundary lines from binarized FMB images
using Canny Edge Detection and the Probabilistic Hough Line Transform (HoughLinesP).
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


class LineDetector:
    """Detects cadastral boundary lines from preprocessed binary images.

    Accepts 2D binary NumPy arrays (black lines on white background from Module 1),
    inverts the polarity for OpenCV edge detectors, and computes line coordinates.
    """

    def __init__(
        self,
        rho: float = 1.0,
        theta: float = np.pi / 180.0,
        threshold: int = 50,
        minLineLength: float = 30.0,
        maxLineGap: float = 10.0,
        canny_low: float = 50.0,
        canny_high: float = 150.0,
        canny_aperture: int = 3,
        **kwargs,
    ) -> None:
        """Initialize the line detector with configurable Hough and Canny parameters.

        Args:
            rho: Distance resolution of the accumulator in pixels (default: 1.0).
            theta: Angle resolution of the accumulator in radians (default: pi/180).
            threshold: Accumulator threshold parameter; only lines with enough votes are returned.
            minLineLength: Minimum line length; line segments shorter than this are rejected.
            maxLineGap: Maximum allowed gap between points on the same line to link them.
            canny_low: Lower hysteresis threshold for Canny edge detector (default: 50.0).
            canny_high: Upper hysteresis threshold for Canny edge detector (default: 150.0).
            canny_aperture: Aperture size for the Sobel operator in Canny (default: 3).
            **kwargs: Allows alternative naming conventions (e.g. min_line_length, max_line_gap).
        """
        self.rho = float(rho)
        self.theta = float(theta)
        self.threshold = int(threshold)

        # Support both camelCase (from prompt) and snake_case (PEP8)
        self.min_line_length = float(kwargs.get("min_line_length", minLineLength))
        self.max_line_gap = float(kwargs.get("max_line_gap", maxLineGap))

        self.canny_low = float(canny_low)
        self.canny_high = float(canny_high)
        self.canny_aperture = int(canny_aperture)

    def detect(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect line segments from a preprocessed binary image.

        Processing Pipeline:
            1. Validate input dimensions and data type.
            2. Invert polarity (Module 1 outputs black lines on white paper;
               OpenCV algorithms expect white foreground on black background).
            3. Apply Canny Edge Detection to extract boundary line gradients.
            4. Apply Probabilistic Hough Line Transform (cv2.HoughLinesP).
            5. Convert output to a list of (x1, y1, x2, y2) tuples.

        Args:
            binary_image: 2D uint8 NumPy array from Module 1 (black lines = 0, white = 255).

        Returns:
            List[Tuple[int, int, int, int]]: List of line coordinates [(x1, y1, x2, y2), ...].
                                            Returns an empty list [] if no lines are detected.

        Raises:
            ValueError: If input image is empty or not a 2D array.
            TypeError: If input is not a numpy.ndarray.
        """
        self._validate_input(binary_image)

        # Step 1: Invert image (Black lines on white bg -> White lines on black bg)
        inverted = cv2.bitwise_not(binary_image)

        # Step 2: Canny Edge Detection
        edges = cv2.Canny(
            inverted,
            threshold1=self.canny_low,
            threshold2=self.canny_high,
            apertureSize=self.canny_aperture,
        )

        # Step 3: Probabilistic Hough Line Transform
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        # Step 4: Handle None case (no lines detected)
        if hough_lines is None:
            return []

        # Step 5: Format into list of (x1, y1, x2, y2) tuples
        # Handles both OpenCV 4.x (N, 1, 4) and OpenCV 5.x (N, 4) shapes
        lines: List[Tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in hough_lines.reshape(-1, 4):
            lines.append((int(x1), int(y1), int(x2), int(y2)))

        return lines

    @staticmethod
    def draw_lines(
        image: np.ndarray,
        lines: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Render detected lines onto an image for visualization or Web GIS overlays.

        Args:
            image: Base image array (grayscale or BGR).
            lines: List of (x1, y1, x2, y2) coordinate tuples.
            color: Line color in BGR format (default: Red (0, 0, 255)).
            thickness: Line thickness in pixels (default: 2).

        Returns:
            np.ndarray: BGR image with rendered line segments.
        """
        if image.ndim == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        for x1, y1, x2, y2 in lines:
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        return canvas

    @staticmethod
    def _validate_input(image: np.ndarray) -> None:
        """Validate input binary array."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
        if image.size == 0:
            raise ValueError("Input binary image is empty.")
        if image.ndim != 2:
            raise ValueError(f"Input binary image must be a 2D array, got {image.ndim} dimensions.")
