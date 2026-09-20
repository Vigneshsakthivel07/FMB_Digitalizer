"""Backend Survey & Comparison Orchestrator (survey_service.py).

Bridges Module 6 (Field Survey) and Module 7 (Spatial Comparison)
with the Database layer and REST API.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from database.repository import GISRepository
from modules.common.models import (
    CadastralPolygon,
    FieldSurveyPolygon,
    PolygonVertex,
    Point2D,
    SpatialComparisonResult,
)
from modules.module6_survey import FieldSurveyManager
from modules.module7_comparison import SpatialComparator


def process_ground_survey(
    parcel_id: int,
    points: List[Dict[str, Any]],
    anchor_fmb_indices: Optional[Tuple[int, int]] = (0, 1),
    anchor_survey_indices: Optional[Tuple[int, int]] = (0, 1),
    surveyor_name: Optional[str] = "Licensed Surveyor",
    survey_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest ground survey points, align to FMB parcel, save to DB, and compare.

    Args:
        parcel_id: Database ID of the FMB parcel.
        points: List of survey points (e.g. [{'point_id':'P1', 'x':..., 'y':...}, ...]).
        anchor_fmb_indices: (idx1, idx2) corner indices in FMB polygon.
        anchor_survey_indices: (idx1, idx2) point indices in survey.
        surveyor_name: Name of surveyor.
        survey_date: Date string.

    Returns:
        Dict with survey polygon, spatial comparison result, and DB IDs.
    """
    repo = GISRepository()
    parcel_record = repo.get_parcel(parcel_id)
    if not parcel_record:
        raise ValueError(f"FMB Parcel #{parcel_id} not found in database.")

    # Reconstruct CadastralPolygon from parcel record
    fmb_polygon = _reconstruct_fmb_polygon_from_record(parcel_record)

    # Module 6: Create and georeference survey polygon
    survey_manager = FieldSurveyManager()
    survey_poly = survey_manager.create_survey_polygon(
        points=points,
        fmb_polygon=fmb_polygon,
        anchor_fmb_indices=anchor_fmb_indices,
        anchor_survey_indices=anchor_survey_indices,
        survey_date=survey_date,
        surveyor_name=surveyor_name,
        parcel_id=parcel_id,
    )

    # Save Field Survey to Database
    survey_id = repo.save_field_survey(survey_poly)
    survey_poly.survey_id = survey_id

    # Module 7: Compare FMB vs Survey
    comparator = SpatialComparator()
    comparison_result = comparator.compare(
        fmb_polygon=fmb_polygon,
        survey_polygon=survey_poly,
        fmb_parcel_id=parcel_id,
    )
    comparison_result.survey_id = survey_id

    # Save Comparison to Database
    comparison_id = repo.save_spatial_comparison(comparison_result)

    return {
        "success": True,
        "parcel_id": parcel_id,
        "survey_id": survey_id,
        "comparison_id": comparison_id,
        "db_backend": repo.backend,
        "survey": survey_poly.to_dict(),
        "comparison": comparison_result.to_dict(),
    }


def _reconstruct_fmb_polygon_from_record(record: Dict[str, Any]) -> CadastralPolygon:
    """Build a CadastralPolygon from database row."""
    vertices = []
    survey_no = record.get("survey_no")
    scale_denom = record.get("scale_denominator") or 848

    # Check geojson_data
    geojson = record.get("geojson_data")
    if isinstance(geojson, str):
        try:
            geojson = json.loads(geojson)
        except Exception:
            geojson = {}

    coords = []
    if geojson and isinstance(geojson, dict):
        if geojson.get("type") == "Feature" and geojson.get("geometry", {}).get("type") == "Polygon":
            coords = geojson["geometry"]["coordinates"][0]
        elif geojson.get("type") == "FeatureCollection":
            for f in geojson.get("features", []):
                geom = f.get("geometry", {})
                if geom.get("type") == "Polygon":
                    coords = geom.get("coordinates", [[]])[0]
                    break
                elif geom.get("type") == "Point" and f.get("properties", {}).get("type") == "vertex":
                    pt = geom.get("coordinates", [0, 0])
                    coords.append(pt)

    # If WKT is present and coords not parsed
    if not coords and record.get("polygon_wkt"):
        wkt = record["polygon_wkt"]
        if "POLYGON" in wkt:
            inner = wkt.split("((")[-1].split("))")[0]
            for pair in inner.split(","):
                parts = pair.strip().split()
                if len(parts) >= 2:
                    coords.append([float(parts[0]), float(parts[1])])

    # De-duplicate consecutive closed ring coordinate
    if len(coords) > 3 and coords[0] == coords[-1]:
        coords = coords[:-1]

    for i, pt in enumerate(coords):
        vx = PolygonVertex(
            index=i,
            pixel=Point2D(pt[0], pt[1]),
            real_world=(float(pt[0]), float(pt[1])),
            is_anchor=(i in (0, 1)),
        )
        vertices.append(vx)

    return CadastralPolygon(
        vertices=vertices,
        area_sqm=record.get("area_sqm"),
        perimeter_m=record.get("perimeter_m"),
        scale_denominator=scale_denom,
        survey_no=survey_no,
        wkt=record.get("polygon_wkt"),
    )

