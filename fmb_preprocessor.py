"""Module 1: FMB Image Preprocessor (fmb_preprocessor.py).

Strictly handles pixel cleaning for FMB survey maps.
Converts raw images into clean, binarized NumPy arrays (black lines on white background)
without detecting lines or geometry.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np


class FMBPreprocessor:
    """Image preprocessor for Field Measurement Book (FMB) survey maps.

    Transforms raw scanned or photographed map images into a clean binarized
    NumPy array with black cadastral lines on a white background.
    """

    def __init__(
        self,
        blur_kernel_size: Tuple[int, int] = (5, 5),
        blur_sigma_x: float = 0.0,
    ) -> None:
        """Initialize preprocessor with Gaussian blur parameters.

        Args:
            blur_kernel_size: Tuple (w, h) specifying Gaussian blur dimensions. Must be odd positive integers.
            blur_sigma_x: Gaussian kernel standard deviation in X direction.
        """
        if blur_kernel_size[0] % 2 == 0 or blur_kernel_size[1] % 2 == 0:
            raise ValueError(f"Blur kernel size must be odd integers, got {blur_kernel_size}")
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma_x = blur_sigma_x

    def read_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Read an image file from disk using OpenCV.

        Args:
            image_path: Path to the image file (JPG/PNG).

        Returns:
            np.ndarray: Loaded image as a NumPy array.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded as an image.
        """
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            # Fallback for unicode or special character paths
            try:
                raw_bytes = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            except Exception:
                image = None

        if image is None:
            raise ValueError(f"Failed to decode image from path: {path}")

        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert an image NumPy array to 8-bit single-channel grayscale.

        Args:
            image: Input image array (BGR color or grayscale).

        Returns:
            np.ndarray: Grayscale image array.
        """
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
        """Apply Gaussian blur to remove paper texture, grain, and scanner noise.

        Args:
            gray_image: Single-channel grayscale image array.

        Returns:
            np.ndarray: Blurred image array.
        """
        return cv2.GaussianBlur(
            gray_image,
            self.blur_kernel_size,
            sigmaX=self.blur_sigma_x,
        )

    def apply_threshold(self, blurred_image: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding to create a binary image (black lines on white background).

        In standard Otsu thresholding (cv2.THRESH_BINARY + cv2.THRESH_OTSU),
        dark ink pixels (< threshold) become 0 (black lines),
        and bright paper background (> threshold) becomes 255 (white background).

        Args:
            blurred_image: Filtered grayscale image array.

        Returns:
            np.ndarray: Binary image array (uint8, values 0 and 255).
        """
        _, binary = cv2.threshold(
            blurred_image,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        return binary

    def process(self, image_path: Union[str, Path]) -> np.ndarray:
        """Execute full preprocessing pipeline on an image file path.

        Steps:
            1. Read image using OpenCV.
            2. Convert to grayscale.
            3. Apply Gaussian blur to remove noise.
            4. Apply Otsu's thresholding (black lines on white background).

        Args:
            image_path: Path to the raw input image file.

        Returns:
            np.ndarray: Clean, binarized NumPy array.
        """
        raw = self.read_image(image_path)
        gray = self.to_grayscale(raw)
        blurred = self.apply_blur(gray)
        binary = self.apply_threshold(blurred)
        return binary


def _is_gui_available() -> bool:
    """Check if a graphical display server (X11/Wayland) is actively reachable."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import subprocess
        res = subprocess.run(
            ["xdpyinfo"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=0.5,
        )
        return res.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    # Determine sample image path from arguments or default mock
    default_sample = Path(__file__).resolve().parent / "noisy_input.png"
    sample_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_sample

    if not sample_path.exists():
        print(f"Sample image not found at {sample_path}. Creating a quick test image...")
        # Create a basic sample test image with a line on noisy background
        sample_img = np.ones((400, 600, 3), dtype=np.uint8) * 230
        cv2.line(sample_img, (50, 200), (550, 200), (20, 20, 20), 3)
        cv2.polylines(
            sample_img,
            [np.array([[100, 100], [500, 100], [450, 300], [150, 300]], np.int32)],
            isClosed=True,
            color=(20, 20, 20),
            thickness=2,
        )
        # Add noise
        noise = np.random.normal(0, 15, sample_img.shape).astype(np.float32)
        sample_img = np.clip(sample_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(sample_path), sample_img)
        print(f"Created sample image: {sample_path}")

    print(f"Testing FMBPreprocessor with: {sample_path}")
    preprocessor = FMBPreprocessor()
    binary_result = preprocessor.process(sample_path)

    print(f"Result shape: {binary_result.shape}, dtype: {binary_result.dtype}")
    print(f"Unique pixel values: {np.unique(binary_result)} (0 = black lines, 255 = white background)")

    # Save output preview alongside sample
    out_preview = sample_path.with_name(f"{sample_path.stem}_binary_bw.png")
    cv2.imwrite(str(out_preview), binary_result)
    print(f"Saved binary output preview to: {out_preview}")

    # Display result using cv2.imshow when a graphical display server is active
    if _is_gui_available():
        try:
            cv2.imshow("FMB Binary Result (Black Lines on White Background)", binary_result)
            print("Displaying image window via cv2.imshow (press any key to close)...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"cv2.imshow note: GUI display error ({e}).")
    else:
        print("Note: Running without an active X11/GUI display. cv2.imshow skipped; binary output saved to file above.")
