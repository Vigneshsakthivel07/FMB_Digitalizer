"""Module 1: FMB Image Preprocessor (fmb_preprocessor.py).

Strictly handles pixel cleaning for FMB survey maps.
Converts raw images into clean, binarized NumPy arrays (black lines on white background)
without detecting lines or geometry.

Designed as a modular, reusable component ready for:
1. CLI file batch processing.
2. In-memory processing pipelines (Dev A -> Dev B).
3. Web GIS dashboard backends (FastAPI, Flask, Streamlit) accepting uploaded bytes and returning base64/PNG streams.
"""

import base64
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

    def decode_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes (e.g. from Web GIS HTTP upload) into a NumPy array.

        Args:
            image_bytes: Raw binary bytes of the image.

        Returns:
            np.ndarray: Loaded image as a NumPy array.

        Raises:
            ValueError: If bytes cannot be decoded as an image.
        """
        if not image_bytes:
            raise ValueError("Provided image bytes are empty.")
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from binary bytes.")
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

    def process_array(self, image: np.ndarray) -> np.ndarray:
        """Execute preprocessing pipeline on an existing NumPy array in memory.

        Args:
            image: Raw NumPy image array.

        Returns:
            np.ndarray: Clean, binarized NumPy array.
        """
        gray = self.to_grayscale(image)
        blurred = self.apply_blur(gray)
        binary = self.apply_threshold(blurred)
        return binary

    def process_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Execute preprocessing pipeline on raw image bytes (ideal for Web GIS uploads).

        Args:
            image_bytes: Raw binary image bytes.

        Returns:
            np.ndarray: Clean, binarized NumPy array.
        """
        raw = self.decode_bytes(image_bytes)
        return self.process_array(raw)

    def process(self, source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """Execute full preprocessing pipeline on a file path, raw bytes, or NumPy array.

        Steps:
            1. Read / decode input into image array.
            2. Convert to grayscale.
            3. Apply Gaussian blur to remove noise.
            4. Apply Otsu's thresholding (black lines on white background).

        Args:
            source: Image file path, binary bytes, or NumPy array.

        Returns:
            np.ndarray: Clean, binarized NumPy array.
        """
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
        """Encode binary image array into PNG bytes (for Web API streaming).

        Args:
            binary_image: Binarized NumPy array.

        Returns:
            bytes: Encoded PNG binary bytes.
        """
        success, encoded = cv2.imencode(".png", binary_image)
        if not success:
            raise RuntimeError("Failed to encode image to PNG format.")
        return encoded.tobytes()

    @staticmethod
    def to_base64_data_uri(binary_image: np.ndarray) -> str:
        """Encode binary image into a base64 Data URI (for Web GIS frontend overlays).

        Args:
            binary_image: Binarized NumPy array.

        Returns:
            str: Data URI string (e.g. 'data:image/png;base64,...').
        """
        png_bytes = FMBPreprocessor.to_png_bytes(binary_image)
        b64_str = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64_str}"


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
    # Determine sample image path from arguments or default sample
    project_root = Path(__file__).resolve().parent
    default_sample = project_root / "samples" / "noisy_input.png"
    if not default_sample.exists():
        default_sample = project_root / "samples" / "real_fmb_input-1.png"

    sample_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_sample

    if not sample_path.exists():
        print(f"Sample image not found at {sample_path}. Creating fallback sample...")
        (project_root / "samples").mkdir(exist_ok=True)
        sample_img = np.ones((400, 600, 3), dtype=np.uint8) * 230
        cv2.line(sample_img, (50, 200), (550, 200), (20, 20, 20), 3)
        cv2.imwrite(str(sample_path), sample_img)

    print(f"Testing FMBPreprocessor with: {sample_path}")
    preprocessor = FMBPreprocessor()
    binary_result = preprocessor.process(sample_path)

    print(f"Result shape: {binary_result.shape}, dtype: {binary_result.dtype}")
    print(f"Unique pixel values: {np.unique(binary_result)} (0 = black lines, 255 = white background)")

    # Save output preview to outputs/ directory
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    out_preview = output_dir / f"{sample_path.stem}_binary_bw.png"
    cv2.imwrite(str(out_preview), binary_result)
    print(f"Saved binary output preview to: {out_preview}")

    # Test Web GIS Dashboard export (Base64 Data URI)
    data_uri = preprocessor.to_base64_data_uri(binary_result)
    print(f"Generated Web GIS Base64 Data URI (length: {len(data_uri)} chars)")

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
        print("Note: Running without an active X11/GUI display. Output saved cleanly to disk.")
