"""PostgreSQL + PostGIS Spatial Repository.

Generates native PostGIS geometry SQL statements (ST_GeomFromText, ST_MakeEnvelope)
and manages database persistence.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from modules.common.models import LineSegment2D, ParcelStructure, Point2D


@dataclass
class PostGISConfig:
    """PostgreSQL / PostGIS connection parameters."""

    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "postgres")
    database: str = os.getenv("PGDATABASE", "fmb_cadastre")
    srid: int = int(os.getenv("PGSRID", "4326"))


class PostGISRepository:
    """Manages geospatial operations and queries against PostgreSQL/PostGIS database."""

    def __init__(self, config: Optional[PostGISConfig] = None):
        self.config = config or PostGISConfig()

    def generate_postgis_sql(
        self,
        parcel: ParcelStructure,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        meta = metadata or {}
        survey_no = parcel.survey_no or meta.get("survey_no", "6/10A")
        village = meta.get("village", "Thalakulam [10]").replace("'", "''")
        taluk = meta.get("taluk", "Bhavani").replace("'", "''")
        district = meta.get("district", "Erode").replace("'", "''")
        scale = meta.get("scale", "1:848")
        srid = self.config.srid

        lines_wkt_parts = [
            f"({l.start.x} {l.start.y}, {l.end.x} {l.end.y})" for l in parcel.lines
        ]
        lines_wkt = (
            f"MULTILINESTRING({', '.join(lines_wkt_parts)})"
            if lines_wkt_parts
            else "MULTILINESTRING EMPTY"
        )

        points_wkt_parts = [f"{c.x} {c.y}" for c in parcel.corners]
        points_wkt = (
            f"MULTIPOINT({', '.join(points_wkt_parts)})"
            if points_wkt_parts
            else "MULTIPOINT EMPTY"
        )

        all_x = [p.x for p in parcel.corners] + [l.start.x for l in parcel.lines] + [l.end.x for l in parcel.lines]
        all_y = [p.y for p in parcel.corners] + [l.start.y for l in parcel.lines] + [l.end.y for l in parcel.lines]
        min_x = min(all_x) if all_x else 0.0
        min_y = min(all_y) if all_y else 0.0
        max_x = max(all_x) if all_x else float(parcel.image_width)
        max_y = max(all_y) if all_y else float(parcel.image_height)

        meta_json = json.dumps(meta).replace("'", "''")

        sql = f"""-- Insert Cadastral Parcel with PostGIS Geometries
WITH new_parcel AS (
    INSERT INTO cadastral_parcels (
        survey_no, village, taluk, district, scale,
        total_lines, total_corners, image_width, image_height,
        lines_geom, vertices_geom, bbox_geom, metadata
    ) VALUES (
        '{survey_no}', '{village}', '{taluk}', '{district}', '{scale}',
        {len(parcel.lines)}, {len(parcel.corners)}, {parcel.image_width}, {parcel.image_height},
        ST_GeomFromText('{lines_wkt}', {srid}),
        ST_GeomFromText('{points_wkt}', {srid}),
        ST_MakeEnvelope({min_x}, {min_y}, {max_x}, {max_y}, {srid}),
        '{meta_json}'::jsonb
    ) RETURNING id
)
"""
        if parcel.lines:
            line_values = []
            for idx, l in enumerate(parcel.lines, start=1):
                line_wkt = f"LINESTRING({l.start.x} {l.start.y}, {l.end.x} {l.end.y})"
                line_values.append(
                    f"((SELECT id FROM new_parcel), '{survey_no}', 'SEG_{idx:03d}', {l.length_px}, {l.angle_deg}, 'boundary', ST_GeomFromText('{line_wkt}', {srid}))"
                )

            sql += f"""INSERT INTO cadastral_boundary_lines (
    parcel_id, survey_no, segment_id, length_px, angle_deg, segment_type, geom
) VALUES
{',\\n'.join(line_values)};\n"""

        if parcel.corners:
            vertex_values = []
            for idx, c in enumerate(parcel.corners, start=1):
                pt_wkt = f"POINT({c.x} {c.y})"
                vertex_values.append(
                    f"((SELECT id FROM new_parcel), '{survey_no}', {idx}, 'corner', ST_GeomFromText('{pt_wkt}', {srid}))"
                )

            sql += f"""INSERT INTO cadastral_vertices (
    parcel_id, survey_no, vertex_index, vertex_type, geom
) VALUES
{',\\n'.join(vertex_values)};\n"""

        return sql

    def export_sql_dump(
        self,
        parcel: ParcelStructure,
        metadata: Optional[Dict[str, Any]] = None,
        output_file: Optional[Union[str, Path]] = None,
    ) -> str:
        schema_path = Path(__file__).resolve().parent / "schema_postgis.sql"
        schema_sql = ""
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()

        insert_sql = self.generate_postgis_sql(parcel, metadata)
        full_sql = f"{schema_sql}\n\n-- Data Ingestion\n{insert_sql}"

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_sql)

        return full_sql

