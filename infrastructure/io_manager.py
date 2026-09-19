"""Infrastructure layer handling file system image I/O operations."""

import os
from pathlib import Path
from typing import Union
import cv2
import numpy as np

from core.interfaces import ImageIOInterface


class LocalFileIOManager(ImageIOInterface):
    """Local file system implementation of ImageIOInterface.

    Decoupled from application logic, enabling easy substitution with
    cloud storage (e.g. S3, GCS) or database adapters in future extensions.
    """

    def load_image(self, source: Union[str, Path]) -> np.ndarray:
        """Load an image from local disk into a NumPy array.

        Args:
            source: File system path to the image file.

        Returns:
            np.ndarray: Loaded image as a NumPy array (BGR or single-channel).

        Raises:
            FileNotFoundError: If the source path does not exist or is not a file.
            ValueError: If the file cannot be decoded as an image.
        """
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found at: {path}")
        if not path.is_file():
            raise ValueError(f"Specified path is not a file: {path}")

        # cv2.imread handles standard file reading
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        # Fallback for unicode/special character paths in various OS environments
        if image is None:
            try:
                raw_bytes = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
            except Exception:
                image = None

        if image is None:
            raise ValueError(f"Failed to decode image file (corrupted or unsupported format): {path}")

        return image

    def save_image(self, destination: Union[str, Path], image: np.ndarray) -> bool:
        """Save an image NumPy array to a local file path.

        Args:
            destination: Target file system path where the image will be written.
            image: Image array to write.

        Returns:
            bool: True if writing succeeded.

        Raises:
            ValueError: If the provided image array is invalid or empty.
            IOError: If OpenCV fails to encode and write the image file.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected image to be a numpy.ndarray, got {type(image)}")
        if image.size == 0:
            raise ValueError("Cannot save an empty image array.")

        dest_path = Path(destination).expanduser().resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(dest_path), image)

        # Fallback for unicode paths if imwrite returns False
        if not success:
            ext = dest_path.suffix.lower() or ".png"
            is_encoded, encoded_img = cv2.imencode(ext, image)
            if is_encoded:
                try:
                    with open(dest_path, "wb") as f:
                        f.write(encoded_img.tobytes())
                    success = True
                except Exception as e:
                    raise IOError(f"Failed to write image to {dest_path}: {e}") from e

        if not success:
            raise IOError(f"OpenCV failed to write image to destination: {dest_path}")

        return True

