"""Module 2: Cadastral Line Detector (line_detector.py).

Detects discrete linear segments and cadastral boundary lines from binarized FMB images
using Canny Edge Detection and Probabilistic Hough Line Transform (HoughLinesP).
"""

from typing import List, Tuple
import cv2
import numpy as np


class LineDetector:
    """Detects cadastral boundary lines from preprocessed binary images."""

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
        self.rho = float(rho)
        self.theta = float(theta)
        self.threshold = int(threshold)
        self.min_line_length = float(kwargs.get("min_line_length", minLineLength))
        self.max_line_gap = float(kwargs.get("max_line_gap", maxLineGap))
        self.canny_low = float(canny_low)
        self.canny_high = float(canny_high)
        self.canny_aperture = int(canny_aperture)

    def detect(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        self._validate_input(binary_image)

        # 1. Invert image (Black lines on white bg -> White lines on black bg)
        inverted = cv2.bitwise_not(binary_image)

        # 2. Canny Edge Detection
        edges = cv2.Canny(
            inverted,
            threshold1=self.canny_low,
            threshold2=self.canny_high,
            apertureSize=self.canny_aperture,
        )

        # 3. Probabilistic Hough Line Transform
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if hough_lines is None:
            return []

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
        if image.ndim == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        for x1, y1, x2, y2 in lines:
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        return canvas

    @staticmethod
    def _validate_input(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
        if image.size == 0:
            raise ValueError("Input binary image is empty.")
        if image.ndim != 2:
            raise ValueError(f"Input binary image must be a 2D array, got {image.ndim} dimensions.")

