"""Unit tests for PostGIS spatial SQL generation and repository layer."""

import tempfile
import unittest
from pathlib import Path

from core.models import LineSegment2D, ParcelStructure, Point2D
from infrastructure.postgis_repository import PostGISConfig, PostGISRepository


class TestPostGISRepository(unittest.TestCase):
    """Test PostGIS SQL formatting, spatial types, and dump exports."""

    def setUp(self):
        self.repo = PostGISRepository(PostGISConfig(srid=4326))
        self.parcel = ParcelStructure(
            lines=[
                LineSegment2D(Point2D(100.0, 200.0), Point2D(400.0, 200.0)),
                LineSegment2D(Point2D(400.0, 200.0), Point2D(400.0, 500.0)),
            ],
            corners=[Point2D(100.0, 200.0), Point2D(400.0, 200.0), Point2D(400.0, 500.0)],
            image_width=600,
            image_height=600,
            survey_no="6/10A",
        )

    def test_generate_postgis_sql(self):
        sql = self.repo.generate_postgis_sql(
            self.parcel,
            metadata={"village": "Thalakulam", "taluk": "Bhavani", "district": "Erode"},
        )
        self.assertIn("ST_GeomFromText('MULTILINESTRING", sql)
        self.assertIn("ST_GeomFromText('MULTIPOINT", sql)
        self.assertIn("ST_MakeEnvelope", sql)
        self.assertIn("cadastral_parcels", sql)
        self.assertIn("cadastral_boundary_lines", sql)
        self.assertIn("cadastral_vertices", sql)
        self.assertIn("4326", sql)

    def test_export_sql_dump(self):
        with tempfile.NamedTemporaryFile(suffix=".sql", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            dump = self.repo.export_sql_dump(
                self.parcel,
                metadata={"survey_no": "6/10A"},
                output_file=tmp_path,
            )
            self.assertIn("CREATE EXTENSION IF NOT EXISTS postgis;", dump)
            self.assertIn("CREATE TABLE IF NOT EXISTS cadastral_parcels", dump)
            self.assertTrue(tmp_path.exists())
            self.assertGreater(tmp_path.stat().st_size, 0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()

