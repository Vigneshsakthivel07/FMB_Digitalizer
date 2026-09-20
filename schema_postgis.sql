-- PostgreSQL + PostGIS Schema for FMB Cadastral GIS Database
-- Enable PostGIS spatial extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- 1. Cadastral Parcels Table
CREATE TABLE IF NOT EXISTS cadastral_parcels (
    id SERIAL PRIMARY KEY,
    survey_no VARCHAR(64) NOT NULL,
    village VARCHAR(128) DEFAULT 'Unknown',
    taluk VARCHAR(128) DEFAULT 'Unknown',
    district VARCHAR(128) DEFAULT 'Unknown',
    scale VARCHAR(32) DEFAULT '1:848',
    total_lines INTEGER NOT NULL,
    total_corners INTEGER NOT NULL,
    image_width INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    -- PostGIS Spatial Geometries (SRID 4326 / WGS84 or 32643 / UTM 43N)
    geom GEOMETRY(Geometry, 4326),
    lines_geom GEOMETRY(MultiLineString, 4326),
    vertices_geom GEOMETRY(MultiPoint, 4326),
    bbox_geom GEOMETRY(Polygon, 4326),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Cadastral Boundary Line Segments Table
CREATE TABLE IF NOT EXISTS cadastral_boundary_lines (
    id SERIAL PRIMARY KEY,
    parcel_id INTEGER NOT NULL REFERENCES cadastral_parcels(id) ON DELETE CASCADE,
    survey_no VARCHAR(64) NOT NULL,
    segment_id VARCHAR(32),
    length_px DOUBLE PRECISION NOT NULL,
    length_meters DOUBLE PRECISION,
    angle_deg DOUBLE PRECISION NOT NULL,
    segment_type VARCHAR(64) DEFAULT 'boundary',
    geom GEOMETRY(LineString, 4326) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Cadastral Corner Vertices Table
CREATE TABLE IF NOT EXISTS cadastral_vertices (
    id SERIAL PRIMARY KEY,
    parcel_id INTEGER NOT NULL REFERENCES cadastral_parcels(id) ON DELETE CASCADE,
    survey_no VARCHAR(64) NOT NULL,
    vertex_index INTEGER,
    vertex_type VARCHAR(64) DEFAULT 'corner',
    geom GEOMETRY(Point, 4326) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Spatial GIST Indexes for sub-millisecond GIS spatial queries
CREATE INDEX IF NOT EXISTS idx_parcels_geom ON cadastral_parcels USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_parcels_lines_geom ON cadastral_parcels USING GIST (lines_geom);
CREATE INDEX IF NOT EXISTS idx_parcels_vertices_geom ON cadastral_parcels USING GIST (vertices_geom);
CREATE INDEX IF NOT EXISTS idx_lines_geom ON cadastral_boundary_lines USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_vertices_geom ON cadastral_vertices USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_parcels_survey_no ON cadastral_parcels (survey_no);

-- Spatial View: Export standard GeoJSON FeatureCollection directly from PostGIS
CREATE OR REPLACE VIEW vw_cadastral_parcels_geojson AS
SELECT 
    id,
    survey_no,
    village,
    taluk,
    district,
    scale,
    total_lines,
    total_corners,
    ST_AsGeoJSON(lines_geom)::json AS lines_geojson,
    ST_AsGeoJSON(vertices_geom)::json AS vertices_geojson,
    ST_AsGeoJSON(bbox_geom)::json AS bbox_geojson,
    created_at
FROM cadastral_parcels;

