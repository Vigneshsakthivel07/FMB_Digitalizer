"""Unit and integration tests for ParcelExtractor, models, and Web pipeline."""

import unittest
from pathlib import Path
import numpy as np
import cv2

from core.models import Point2D, LineSegment2D, ParcelStructure, FMBProcessingResult
from parcel_extractor import FMBLayoutSegmenter, ParcelBoundaryDetector, extract_parcel_from_source
from fmb_pipeline import process_for_web, run_pipeline


class TestParcelModels(unittest.TestCase):
    """Test geometric data models and GeoJSON serialization."""

    def test_point_and_line_segment(self):
        p1 = Point2D(10.0, 20.0)
        p2 = Point2D(40.0, 60.0)
        self.assertEqual(p1.distance_to(p2), 50.0)

        line = LineSegment2D(start=p1, end=p2)
        self.assertEqual(line.length_px, 50.0)
        self.assertEqual(line.to_tuple(), (10, 20, 40, 60))

        geojson_feature = line.to_geojson_feature()
        self.assertEqual(geojson_feature["geometry"]["type"], "LineString")
        self.assertEqual(geojson_feature["geometry"]["coordinates"], [[10.0, 20.0], [40.0, 60.0]])

    def test_parcel_structure_to_geojson(self):
        parcel = ParcelStructure(
            lines=[
                LineSegment2D(Point2D(0, 0), Point2D(100, 0)),
                LineSegment2D(Point2D(100, 0), Point2D(100, 100)),
            ],
            corners=[Point2D(0, 0), Point2D(100, 0), Point2D(100, 100)],
            image_width=200,
            image_height=200,
            survey_no="Test/1",
        )
        geojson = parcel.to_geojson()
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 5)  # 2 lines + 3 points
        self.assertIn("Test/1", geojson["properties"]["survey_no"])


class TestParcelExtractor(unittest.TestCase):
    """Test layout segmentation, text removal, and corner vertex extraction."""

    def setUp(self):
        # Create synthetic canvas with outer border and inner parcel rectangle
        self.canvas = np.ones((500, 500, 3), dtype=np.uint8) * 240
        # Outer document frame (should be filtered out by layout segmenter)
        cv2.rectangle(self.canvas, (10, 10), (490, 490), (0, 0, 0), 2)
        # Inner parcel rectangle (should be detected)
        cv2.rectangle(self.canvas, (150, 150), (350, 350), (0, 0, 0), 3)

    def test_segmenter_with_numpy_and_bytes(self):
        # Test array input
        segmenter = FMBLayoutSegmenter(self.canvas)
        mask = segmenter.get_masked_image()
        self.assertEqual(mask.shape, (500, 500))

        # Test byte input
        _, buf = cv2.imencode(".png", self.canvas)
        segmenter_bytes = FMBLayoutSegmenter(buf.tobytes())
        mask_bytes = segmenter_bytes.get_masked_image()
        self.assertEqual(mask_bytes.shape, (500, 500))

    def test_extract_parcel_from_source(self):
        parcel, outline = extract_parcel_from_source(self.canvas, survey_no="Synthetic/1")
        self.assertIsInstance(parcel, ParcelStructure)
        self.assertGreater(len(parcel.lines), 0)
        self.assertEqual(outline.shape, (500, 500, 3))


class TestWebPipelineIntegration(unittest.TestCase):
    """Test process_for_web interface for web application readiness."""

    def test_process_for_web_returns_valid_result(self):
        canvas = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(canvas, (100, 100), (300, 300), (0, 0, 0), 2)

        result = process_for_web(canvas, survey_no="Web/100")
        self.assertIsInstance(result, FMBProcessingResult)
        self.assertTrue(result.success)
        self.assertTrue(result.binary_preview_base64.startswith("data:image/png;base64,"))
        self.assertTrue(result.overlay_preview_base64.startswith("data:image/png;base64,"))
        self.assertTrue(result.outline_preview_base64.startswith("data:image/png;base64,"))

        geojson = result.to_geojson()
        self.assertEqual(geojson["type"], "FeatureCollection")


if __name__ == "__main__":
    unittest.main()

