"""Module: Parcel Extractor (parcel_extractor.py).

Isolates FMB land parcel boundaries, filters document frame margins/headers/footers,
removes text annotations, and computes polygon corner vertices.
Ready for Web GIS backend services and batch pipelines.
"""

import base64
import math
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from core.models import LineSegment2D, ParcelStructure, Point2D


class FMBLayoutSegmenter:
    """Segments FMB survey sheet layout and isolates central land parcel drawing canvas."""

    def __init__(self, source: Union[str, Path, bytes, np.ndarray]):
        if isinstance(source, (str, Path)):
            self.image_path = str(source)
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                raise ValueError(f"Unable to read image at {source}")
        elif isinstance(source, bytes):
            self.image_path = "in_memory_bytes"
            np_arr = np.frombuffer(source, np.uint8)
            self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if self.image is None:
                raise ValueError("Failed to decode image from provided bytes.")
        elif isinstance(source, np.ndarray):
            self.image_path = "in_memory_array"
            self.image = source.copy()
            if self.image.ndim == 2:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")

        self.height, self.width = self.image.shape[:2]

    def _preprocess(self) -> np.ndarray:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )
        return binary

    def _get_canvas_mask(self) -> np.ndarray:
        binary = self._preprocess()
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_x1 = int(self.width * 0.05)
        roi_y1 = int(self.height * 0.16)
        roi_x2 = int(self.width * 0.95)
        roi_y2 = int(self.height * 0.85)

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > self.width * 0.85 and h > self.height * 0.85:
                continue
            if y + h < roi_y1 or y > roi_y2:
                continue
            cv2.drawContours(mask, [cnt], -1, 255, -1)

        roi_mask = np.zeros_like(mask)
        roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
        canvas_mask = cv2.bitwise_and(binary, roi_mask)

        return canvas_mask

    def get_masked_image(self) -> np.ndarray:
        return self._get_canvas_mask()


class ParcelBoundaryDetector:
    """Detects land parcel line segments and computes corner vertices on masked canvas."""

    def __init__(self, masked_image: np.ndarray):
        self.masked_image = masked_image
        self.height, self.width = masked_image.shape[:2]

    def _remove_text(self) -> np.ndarray:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(self.masked_image, cv2.MORPH_CLOSE, kernel_close)

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

        line_h = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_h)
        line_v = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_v)

        combined_lines = cv2.bitwise_or(line_h, line_v)

        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_lines = cv2.dilate(combined_lines, kernel_connect, iterations=1)

        return clean_lines

    def _detect_lines(self) -> List[Tuple[int, int, int, int]]:
        cleaned = self._remove_text()
        lines = cv2.HoughLinesP(
            cleaned,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=30,
            maxLineGap=20,
        )
        if lines is None:
            return []
        formatted_lines: List[Tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            formatted_lines.append((int(x1), int(y1), int(x2), int(y2)))
        return formatted_lines

    @staticmethod
    def _compute_intersection(
        line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]
    ) -> Optional[Tuple[int, int]]:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if -0.1 <= t <= 1.1 and -0.1 <= u <= 1.1:
            px = int(round(x1 + t * (x2 - x1)))
            py = int(round(y1 + t * (y2 - y1)))
            return px, py
        return None

    def _find_corners(
        self, lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int]]:
        corners: List[Tuple[int, int]] = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self._compute_intersection(lines[i], lines[j])
                if pt:
                    x, y = pt
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if not any(math.hypot(x - cx, y - cy) < 15 for cx, cy in corners):
                            corners.append((x, y))
        return corners

    def detect_parcel_structure(
        self,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int]]]:
        lines = self._detect_lines()
        corners = self._find_corners(lines)
        return lines, corners

    def extract_parcel(self, survey_no: Optional[str] = None) -> ParcelStructure:
        """High-level extraction method returning structured domain model."""
        lines_raw, corners_raw = self.detect_parcel_structure()
        
        line_objects = [
            LineSegment2D(start=Point2D(x1, y1), end=Point2D(x2, y2))
            for x1, y1, x2, y2 in lines_raw
        ]
        corner_objects = [Point2D(x, y) for x, y in corners_raw]

        return ParcelStructure(
            lines=line_objects,
            corners=corner_objects,
            image_width=self.width,
            image_height=self.height,
            survey_no=survey_no,
        )


def extract_parcel_from_source(
    source: Union[str, Path, bytes, np.ndarray],
    survey_no: Optional[str] = None,
) -> Tuple[ParcelStructure, np.ndarray]:
    """Reusable utility function for web handlers and pipelines."""
    segmenter = FMBLayoutSegmenter(source)
    masked = segmenter.get_masked_image()
    detector = ParcelBoundaryDetector(masked)
    parcel = detector.extract_parcel(survey_no=survey_no)

    # Render clean outline image
    clean_canvas = np.ones((segmenter.height, segmenter.width, 3), dtype=np.uint8) * 255
    for line in parcel.lines:
        x1, y1, x2, y2 = line.to_tuple()
        cv2.line(clean_canvas, (x1, y1), (x2, y2), (0, 0, 0), 2, cv2.LINE_AA)
    for corner in parcel.corners:
        x, y = corner.to_tuple()
        cv2.circle(clean_canvas, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(clean_canvas, (x, y), 8, (255, 0, 0), 1, cv2.LINE_AA)

    return parcel, clean_canvas


if __name__ == "__main__":
    sample_path = "samples/user_fmb_input-1.png"
    if len(sys.argv) > 1:
        sample_path = sys.argv[1]
    elif not Path(sample_path).exists():
        sample_path = "samples/real_fmb_input-1.png"

    parcel, clean_canvas = extract_parcel_from_source(sample_path, survey_no="6/10A")
    cv2.imwrite("parcel_outline_only.png", clean_canvas)
    print(f"Extracted {len(parcel.lines)} parcel lines and {len(parcel.corners)} corner vertices.")
    print(f"Saved clean outline to: parcel_outline_only.png")
