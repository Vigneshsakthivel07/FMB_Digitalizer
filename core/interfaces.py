"""Domain interfaces and abstract base classes for FMB image processing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
import numpy as np

from core.config import PreprocessorConfig


class ImageIOInterface(ABC):
    """Abstract interface for image input/output operations.

    Decouples storage implementations (local file system, cloud storage,
    in-memory stream) from domain and application logic.
    """

    @abstractmethod
    def load_image(self, source: Union[str, Path]) -> np.ndarray:
        """Load an image from a given source path or identifier.

        Args:
            source: Path or URI pointing to the image file.

        Returns:
            np.ndarray: Loaded image as a NumPy array (BGR or grayscale).

        Raises:
            FileNotFoundError: If the source does not exist.
            ValueError: If the file cannot be decoded as an image.
        """
        pass

    @abstractmethod
    def save_image(self, destination: Union[str, Path], image: np.ndarray) -> bool:
        """Save an image NumPy array to a destination path or identifier.

        Args:
            destination: Path or URI where the image should be written.
            image: Image array to save.

        Returns:
            bool: True if saving was successful, False otherwise.

        Raises:
            ValueError: If the image array is invalid or cannot be encoded.
            OSError: If destination cannot be written to.
        """
        pass


class ImagePreprocessorInterface(ABC):
    """Abstract interface for image preprocessors."""

    @abstractmethod
    def process(
        self, image: np.ndarray, config: Optional[PreprocessorConfig] = None
    ) -> np.ndarray:
        """Execute preprocessing pipeline on an image array.

        Args:
            image: Input raw image as a NumPy array.
            config: Optional PreprocessorConfig to customize processing parameters.
                    If None, preprocessor default configuration should be used.

        Returns:
            np.ndarray: Cleaned binary image array (uint8).

        Raises:
            ValueError: If input array is invalid, empty, or incompatible.
        """
        pass

