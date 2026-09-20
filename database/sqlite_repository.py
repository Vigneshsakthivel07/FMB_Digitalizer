"""SQLite Local GIS Repository Layer."""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from modules.common.models import LineSegment2D, ParcelStructure, Point2D


class SQLiteGISRepository:
    """Geospatial SQLite repository for storing digitized FMB cadastral parcels."""

    def __init__(self, db_path: str = "fmb_gis.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fmb_parcels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    survey_no TEXT NOT NULL,
                    village TEXT DEFAULT 'Unknown',
                    taluk TEXT DEFAULT 'Unknown',
                    district TEXT DEFAULT 'Unknown',
                    scale TEXT DEFAULT '1:848',
                    total_lines INTEGER NOT NULL,
                    total_corners INTEGER NOT NULL,
                    image_width INTEGER NOT NULL,
                    image_height INTEGER NOT NULL,
                    bbox_min_x REAL,
                    bbox_min_y REAL,
                    bbox_max_x REAL,
                    bbox_max_y REAL,
                    geojson_data TEXT NOT NULL,
                    wkt_geometry TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fmb_lines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parcel_id INTEGER NOT NULL,
                    start_x REAL NOT NULL,
                    start_y REAL NOT NULL,
                    end_x REAL NOT NULL,
                    end_y REAL NOT NULL,
                    length_px REAL NOT NULL,
                    angle_deg REAL NOT NULL,
                    segment_type TEXT DEFAULT 'boundary',
                    FOREIGN KEY (parcel_id) REFERENCES fmb_parcels (id) ON DELETE CASCADE
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fmb_vertices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parcel_id INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    vertex_type TEXT DEFAULT 'corner',
                    FOREIGN KEY (parcel_id) REFERENCES fmb_parcels (id) ON DELETE CASCADE
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_parcels_survey ON fmb_parcels (survey_no);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_lines_parcel ON fmb_lines (parcel_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vertices_parcel ON fmb_vertices (parcel_id);")
            conn.commit()

    def save_parcel(
        self,
        parcel: ParcelStructure,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = metadata or {}
        survey_no = parcel.survey_no or meta.get("survey_no", "6/10A")
        village = meta.get("village", "Thalakulam [10]")
        taluk = meta.get("taluk", "Bhavani")
        district = meta.get("district", "Erode")
        scale = meta.get("scale", "1:848")

        all_x = [p.x for p in parcel.corners] + [l.start.x for l in parcel.lines] + [l.end.x for l in parcel.lines]
        all_y = [p.y for p in parcel.corners] + [l.start.y for l in parcel.lines] + [l.end.y for l in parcel.lines]

        min_x = min(all_x) if all_x else 0.0
        min_y = min(all_y) if all_y else 0.0
        max_x = max(all_x) if all_x else float(parcel.image_width)
        max_y = max(all_y) if all_y else float(parcel.image_height)

        geojson_str = json.dumps(parcel.to_geojson())
        wkt = f"MULTILINESTRING({', '.join([f'({l.start.x} {l.start.y}, {l.end.x} {l.end.y})' for l in parcel.lines])})" if parcel.lines else ""

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO fmb_parcels (
                    survey_no, village, taluk, district, scale,
                    total_lines, total_corners, image_width, image_height,
                    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y,
                    geojson_data, wkt_geometry, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    survey_no,
                    village,
                    taluk,
                    district,
                    scale,
                    len(parcel.lines),
                    len(parcel.corners),
                    parcel.image_width,
                    parcel.image_height,
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                    geojson_str,
                    wkt,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            parcel_id = cursor.lastrowid

            lines_data = [
                (
                    parcel_id,
                    l.start.x,
                    l.start.y,
                    l.end.x,
                    l.end.y,
                    l.length_px,
                    l.angle_deg,
                    "boundary",
                )
                for l in parcel.lines
            ]
            cursor.executemany(
                """
                INSERT INTO fmb_lines (
                    parcel_id, start_x, start_y, end_x, end_y,
                    length_px, angle_deg, segment_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                lines_data,
            )

            vertices_data = [(parcel_id, c.x, c.y, "corner") for c in parcel.corners]
            cursor.executemany(
                """
                INSERT INTO fmb_vertices (parcel_id, x, y, vertex_type)
                VALUES (?, ?, ?, ?)
                """,
                vertices_data,
            )

            conn.commit()
            return parcel_id

    def list_parcels(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, survey_no, village, taluk, district, scale,
                       total_lines, total_corners, image_width, image_height,
                       bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, created_at
                FROM fmb_parcels
                ORDER BY id DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_parcel(self, parcel_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM fmb_parcels WHERE id = ?", (parcel_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["geojson_data"] = json.loads(data["geojson_data"])
            return data

    def get_all_geojson(self) -> Dict[str, Any]:
        parcels = self.list_parcels()
        all_features = []
        for p in parcels:
            rec = self.get_parcel(p["id"])
            if rec and "geojson_data" in rec:
                all_features.extend(rec["geojson_data"].get("features", []))

        return {
            "type": "FeatureCollection",
            "features": all_features,
            "properties": {"total_parcels": len(parcels)},
        }

