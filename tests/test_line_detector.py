"""Unit and integration tests for Module 2: Cadastral Line Detector."""

import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2

from line_detector import LineDetector
from fmb_pipeline import run_pipeline


class TestLineDetectorUnit(unittest.TestCase):
    """Unit tests for LineDetector class."""

    def setUp(self):
        """Initialize detector with standard parameters."""
        self.detector = LineDetector(
            threshold=30,
            minLineLength=40.0,
            maxLineGap=5.0,
        )

    def test_line_detection_horizontal(self):
        """Create a white canvas with a black line (simulating Module 1 output) and detect it."""
        # 1. Simulate Module 1 output: 2D uint8 array with white background (255) and black line (0)
        h, w = 200, 300
        canvas = np.ones((h, w), dtype=np.uint8) * 255
        # Draw a distinct black horizontal line from (50, 100) to (250, 100)
        cv2.line(canvas, (50, 100), (250, 100), color=0, thickness=2)

        # 2. Run detector
        lines = self.detector.detect(canvas)

        # 3. Verify that lines were detected
        self.assertIsInstance(lines, list)
        self.assertGreater(len(lines), 0, "Expected at least one line segment to be detected.")

        # 4. Verify tuple format: (x1, y1, x2, y2)
        for line in lines:
            self.assertEqual(len(line), 4)
            x1, y1, x2, y2 = line
            self.assertIsInstance(x1, int)
            self.assertIsInstance(y1, int)
            self.assertIsInstance(x2, int)
            self.assertIsInstance(y2, int)

            # Check that y-coordinates are centered around y=100 (+/- 2 pixels)
            self.assertAlmostEqual(y1, 100, delta=3)
            self.assertAlmostEqual(y2, 100, delta=3)

    def test_empty_canvas_returns_empty_list(self):
        """A pure white canvas with no lines should return an empty list [] without error."""
        canvas = np.ones((150, 150), dtype=np.uint8) * 255
        lines = self.detector.detect(canvas)
        self.assertEqual(lines, [])

    def test_invalid_inputs_raise_errors(self):
        """Verify proper validation on invalid inputs."""
        # Not a numpy array
        with self.assertRaises(TypeError):
            self.detector.detect("not an array")  # type: ignore

        # Empty array
        with self.assertRaises(ValueError):
            self.detector.detect(np.array([], dtype=np.uint8))

        # 3D array (should be 2D binary)
        with self.assertRaises(ValueError):
            self.detector.detect(np.ones((100, 100, 3), dtype=np.uint8))

    def test_draw_lines(self):
        """Verify line rendering utility function."""
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        lines = [(10, 10, 90, 10)]
        drawn = self.detector.draw_lines(canvas, lines, color=(0, 255, 0), thickness=2)
        self.assertEqual(drawn.shape, (100, 100, 3))
        # Pixel along the drawn line should now be green
        self.assertEqual(drawn[10, 50, 1], 255)


class TestPipelineIntegrationMock(unittest.TestCase):
    """Integration tests verifying fmb_pipeline.run_pipeline with mocks."""

    def test_run_pipeline_with_mock_preprocessor(self):
        """Mock FMBPreprocessor to return a known binary array and verify pipeline flow."""
        # 1. Create mock preprocessor
        mock_preprocessor = MagicMock()
        # Synthetic binary canvas with black cross lines
        mock_binary = np.ones((200, 200), dtype=np.uint8) * 255
        cv2.line(mock_binary, (20, 100), (180, 100), color=0, thickness=2)
        cv2.line(mock_binary, (100, 20), (100, 180), color=0, thickness=2)
        mock_preprocessor.process.return_value = mock_binary

        # 2. Create detector with matching threshold
        detector = LineDetector(threshold=20, minLineLength=30.0, maxLineGap=5.0)

        # 3. Run pipeline
        dummy_input = "mock_fmb_input.jpg"
        result_lines = run_pipeline(dummy_input, preprocessor=mock_preprocessor, detector=detector)

        # 4. Assert preprocessor was called with the exact input source
        mock_preprocessor.process.assert_called_once_with(dummy_input)

        # 5. Assert returned line list format
        self.assertIsInstance(result_lines, list)
        self.assertGreater(len(result_lines), 0)
        for line in result_lines:
            self.assertEqual(len(line), 4)
            self.assertIsInstance(line[0], int)
            self.assertIsInstance(line[1], int)
            self.assertIsInstance(line[2], int)
            self.assertIsInstance(line[3], int)


if __name__ == "__main__":
    unittest.main()
