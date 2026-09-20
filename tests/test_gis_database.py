"""Unit tests for GISDatabase repository layer."""

import os
import tempfile
import unittest
from pathlib import Path

from core.models import LineSegment2D, ParcelStructure, Point2D
from infrastructure.gis_database import GISDatabase


class TestGISDatabase(unittest.TestCase):
    """Test GIS Database SQLite persistence, GeoJSON exports, and querying."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db = GISDatabase(self.temp_db.name)

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_save_and_retrieve_parcel(self):
        parcel = ParcelStructure(
            lines=[
                LineSegment2D(Point2D(100, 100), Point2D(300, 100)),
                LineSegment2D(Point2D(300, 100), Point2D(300, 300)),
            ],
            corners=[Point2D(100, 100), Point2D(300, 100), Point2D(300, 300)],
            image_width=500,
            image_height=500,
            survey_no="6/10A",
        )

        parcel_id = self.db.save_parcel(
            parcel,
            metadata={
                "village": "Thalakulam [10]",
                "taluk": "Bhavani",
                "district": "Erode",
                "scale": "1:848",
            },
        )
        self.assertIsInstance(parcel_id, int)
        self.assertGreater(parcel_id, 0)

        # Retrieve
        record = self.db.get_parcel(parcel_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["survey_no"], "6/10A")
        self.assertEqual(record["total_lines"], 2)
        self.assertEqual(record["total_corners"], 3)
        self.assertIn("features", record["geojson_data"])

    def test_list_parcels_and_export_all(self):
        parcel = ParcelStructure(
            lines=[LineSegment2D(Point2D(0, 0), Point2D(50, 50))],
            corners=[Point2D(0, 0), Point2D(50, 50)],
            image_width=100,
            image_height=100,
            survey_no="Test/2",
        )
        self.db.save_parcel(parcel)

        parcels = self.db.list_parcels()
        self.assertEqual(len(parcels), 1)

        all_geojson = self.db.get_all_geojson()
        self.assertEqual(all_geojson["type"], "FeatureCollection")
        self.assertGreater(len(all_geojson["features"]), 0)


if __name__ == "__main__":
    unittest.main()

