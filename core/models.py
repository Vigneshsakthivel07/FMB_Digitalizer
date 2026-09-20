"""Domain data models and schemas for FMB Cadastral features.

Designed for seamless serialization to GeoJSON, JSON, and Web API responses (FastAPI, Flask, Streamlit).
"""

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Point2D:
    """2D coordinate point representing vertex or marker."""

    x: float
    y: float

    def to_tuple(self) -> Tuple[int, int]:
        return int(round(self.x)), int(round(self.y))

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 2), "y": round(self.y, 2)}

    def distance_to(self, other: "Point2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class LineSegment2D:
    """2D line segment representing a cadastral boundary or subdivision line."""

    start: Point2D
    end: Point2D
    length_px: float = field(init=False)
    angle_deg: float = field(init=False)

    def __post_init__(self):
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx)) % 180
        object.__setattr__(self, "length_px", round(length, 2))
        object.__setattr__(self, "angle_deg", round(angle, 2))

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.start.x)),
            int(round(self.start.y)),
            int(round(self.end.x)),
            int(round(self.end.y)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "length_px": self.length_px,
            "angle_deg": self.angle_deg,
        }

    def to_geojson_feature(self, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [self.start.x, self.start.y],
                    [self.end.x, self.end.y],
                ],
            },
            "properties": properties or {"length_px": self.length_px, "angle_deg": self.angle_deg},
        }


@dataclass
class ParcelStructure:
    """Encapsulates detected cadastral parcel boundaries, vertices, and metadata."""

    lines: List[LineSegment2D] = field(default_factory=list)
    corners: List[Point2D] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    survey_no: Optional[str] = None
    scale_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "survey_no": self.survey_no,
            "image_dimensions": {"width": self.image_width, "height": self.image_height},
            "total_lines": len(self.lines),
            "total_corners": len(self.corners),
            "lines": [line.to_dict() for line in self.lines],
            "corners": [corner.to_dict() for corner in self.corners],
        }

    def to_geojson(self) -> Dict[str, Any]:
        features = []
        # Line features
        for i, line in enumerate(self.lines, start=1):
            features.append(line.to_geojson_feature({"id": f"line_{i}", "type": "boundary_segment"}))
        # Corner point features
        for i, corner in enumerate(self.corners, start=1):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [corner.x, corner.y],
                    },
                    "properties": {"id": f"corner_{i}", "type": "vertex"},
                }
            )
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "survey_no": self.survey_no,
                "image_width": self.image_width,
                "image_height": self.image_height,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class FMBProcessingResult:
    """Standard unified response object for Web GIS APIs and pipeline callers."""

    success: bool
    parcel: ParcelStructure
    binary_preview_base64: Optional[str] = None
    overlay_preview_base64: Optional[str] = None
    outline_preview_base64: Optional[str] = None
    message: str = "Processing completed successfully"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "parcel": self.parcel.to_dict(),
            "previews": {
                "binary": self.binary_preview_base64,
                "overlay": self.overlay_preview_base64,
                "outline": self.outline_preview_base64,
            },
        }

    def to_geojson(self) -> Dict[str, Any]:
        return self.parcel.to_geojson()

