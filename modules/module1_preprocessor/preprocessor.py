"""Module 1: FMB Image Preprocessor (preprocessor.py).

Handles pixel cleaning for FMB survey maps.
Converts raw images into clean, binarized NumPy arrays (black lines on white background).
"""

import base64
import os
from pathlib import Path
import sys
from typing import Tuple, Union
import cv2
import numpy as np


class FMBPreprocessor:
    """Image preprocessor for Field Measurement Book (FMB) survey maps."""

    def __init__(
        self,
        blur_kernel_size: Tuple[int, int] = (5, 5),
        blur_sigma_x: float = 0.0,
    ) -> None:
        if blur_kernel_size[0] % 2 == 0 or blur_kernel_size[1] % 2 == 0:
            raise ValueError(f"Blur kernel size must be odd integers, got {blur_kernel_size}")
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma_x = blur_sigma_x

    def read_image(self, image_path: Union[str, Path]) -> np.ndarray:
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            try:
                raw_bytes = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            except Exception:
                image = None

        if image is None:
            raise ValueError(f"Failed to decode image from path: {path}")

        return image

    def decode_bytes(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("Provided image bytes are empty.")
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from binary bytes.")
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if image.ndim == 3 and image.shape[2] == 1:
            return image[:, :, 0]
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")

    def apply_blur(self, gray_image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            gray_image,
            self.blur_kernel_size,
            sigmaX=self.blur_sigma_x,
        )

    def apply_threshold(self, blurred_image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(
            blurred_image,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        return binary

    def process_array(self, image: np.ndarray) -> np.ndarray:
        gray = self.to_grayscale(image)
        blurred = self.apply_blur(gray)
        binary = self.apply_threshold(blurred)
        return binary

    def process_bytes(self, image_bytes: bytes) -> np.ndarray:
        raw = self.decode_bytes(image_bytes)
        return self.process_array(raw)

    def process(self, source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        if isinstance(source, (str, Path)):
            raw = self.read_image(source)
        elif isinstance(source, bytes):
            raw = self.decode_bytes(source)
        elif isinstance(source, np.ndarray):
            raw = source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        return self.process_array(raw)

    @staticmethod
    def to_png_bytes(binary_image: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(".png", binary_image)
        if not success:
            raise RuntimeError("Failed to encode image to PNG format.")
        return encoded.tobytes()

    @staticmethod
    def to_base64_data_uri(binary_image: np.ndarray) -> str:
        png_bytes = FMBPreprocessor.to_png_bytes(binary_image)
        b64_str = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64_str}"

