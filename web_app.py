"""FMB Digitizer Web Application & GIS Server.

Provides:
- Interactive Multi-Stage Image & Vector Studio UI.
- REST API for FMB digitization, line detection, and layout segmentation.
- SQLite GIS database integration for storing and querying cadastral parcels.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.parse
import cv2
import numpy as np

from core.models import FMBProcessingResult, ParcelStructure
from fmb_pipeline import process_for_web
from infrastructure.gis_database import GISDatabase
from infrastructure.postgis_repository import PostGISRepository

PORT = 8080
gis_db = GISDatabase("fmb_gis.db")
postgis_repo = PostGISRepository()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Tamil Nadu Cadastral FMB Digitizer & GIS Hub</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --border: #334155;
      --success: #22c55e;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #090d16;
      border-bottom: 1px solid var(--border);
      padding: 1rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    header h1 { font-size: 1.25rem; font-weight: 700; color: var(--accent); display: flex; align-items: center; gap: 0.5rem; }
    .badge { background: #0369a1; color: #e0f2fe; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; }
    
    .nav-tabs { display: flex; gap: 1rem; }
    .nav-tab {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-weight: 600;
      padding: 0.5rem 1rem;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
    }
    .nav-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
    
    main { padding: 2rem; flex: 1; max-width: 1400px; margin: 0 auto; width: 100%; }
    
    .view-panel { display: none; }
    .view-panel.active { display: block; }
    
    /* Upload Section */
    .hero-card {
      background: var(--card-bg);
      border: 1px dashed var(--border);
      border-radius: 12px;
      padding: 2.5rem;
      text-align: center;
      margin-bottom: 2rem;
    }
    .btn {
      background: var(--accent);
      color: #0f172a;
      font-weight: 600;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }
    .btn:hover { background: var(--accent-hover); }
    .btn-secondary {
      background: #334155;
      color: var(--text);
      margin-left: 0.5rem;
    }
    .btn-secondary:hover { background: #475569; }
    .btn-success { background: var(--success); color: #0f172a; }
    
    /* Metrics Row */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem;
    }
    .metric-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: var(--accent); margin-top: 0.25rem; }
    
    /* Stage Comparison Viewer */
    .stages-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1.5rem;
      margin-bottom: 2rem;
    }
    .stage-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .stage-header {
      padding: 0.75rem 1rem;
      background: #111827;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .stage-img-container {
      flex: 1;
      min-height: 380px;
      max-height: 480px;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: auto;
      padding: 0.5rem;
    }
    .stage-img-container img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: 4px;
    }
    
    /* GIS Table */
    .table-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }
    table { width: 100%; border-collapse: collapse; text-align: left; }
    th { background: #111827; padding: 1rem; font-size: 0.85rem; color: var(--text-muted); border-bottom: 1px solid var(--border); }
    td { padding: 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
    tr:hover { background: #243248; }

    #loading { display: none; margin: 1rem 0; font-weight: 600; color: var(--accent); }
    .spinner { display: inline-block; width: 1rem; height: 1rem; border: 2px solid var(--accent); border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <header>
    <h1>🗺️ FMB Cadastral Digitizer <span class="badge">Module 1 + 2 + GIS DB</span></h1>
    <div class="nav-tabs">
      <button class="nav-tab active" onclick="switchTab('studio')">Digitizer Studio</button>
      <button class="nav-tab" onclick="switchTab('database')">GIS Database Explorer</button>
    </div>
  </header>

  <main>
    <!-- TAB 1: STUDIO -->
    <div id="tab-studio" class="view-panel active">
      <div class="hero-card">
        <h2>Process Field Measurement Book (FMB) Maps</h2>
        <p style="color: var(--text-muted); margin: 0.5rem 0 1.5rem 0;">
          Upload raw scanned FMB image (or load sample) to isolate cadastral boundaries, remove document neatlines, detect vertices, and store into GIS.
        </p>
        <input type="file" id="fileInput" accept="image/*,.pdf" style="display:none;" onchange="handleFileUpload(event)">
        <button class="btn" onclick="document.getElementById('fileInput').click()">📁 Upload FMB Map</button>
        <button class="btn btn-secondary" onclick="loadSampleFMB()">⚡ Run on Sample (Survey 6/10A)</button>
        <div id="loading"><span class="spinner"></span> Processing through Module 1, Module 2, and Layout Extractor...</div>
      </div>

      <!-- Metrics Row -->
      <div id="metricsRow" class="metrics-grid" style="display:none;">
        <div class="metric-card">
          <div class="metric-label">Survey Identifier</div>
          <div class="metric-val" id="metricSurvey">6/10A</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Detected Lines</div>
          <div class="metric-val" id="metricLines">0</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Corner Vertices</div>
          <div class="metric-val" id="metricCorners">0</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">GIS Status</div>
          <div class="metric-val" style="color:var(--success); font-size:1.1rem; display:flex; align-items:center; gap:0.4rem;">
            ✓ Saved in SQLite GIS
          </div>
        </div>
      </div>

      <!-- Stages Visual Grid -->
      <div id="stagesRow" class="stages-grid" style="display:none;">
        <div class="stage-card">
          <div class="stage-header"><span>1. Mod 1: Binarized (Otsu)</span></div>
          <div class="stage-img-container"><img id="imgBinary" src="" alt="Binary"></div>
        </div>
        <div class="stage-card">
          <div class="stage-header"><span>2. Mod 2: Line Detector (Red)</span></div>
          <div class="stage-img-container"><img id="imgOverlay" src="" alt="Overlay"></div>
        </div>
        <div class="stage-card">
          <div class="stage-header">
            <span>3. Isolated Parcel Outline</span>
            <button class="btn btn-secondary" style="padding:0.2rem 0.5rem; font-size:0.75rem;" onclick="downloadGeoJSON()">📥 GeoJSON</button>
          </div>
          <div class="stage-img-container"><img id="imgOutline" src="" alt="Parcel Outline"></div>
        </div>
      </div>
    </div>

    <!-- TAB 2: GIS DATABASE -->
    <div id="tab-database" class="view-panel">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;">
        <h2>Stored Cadastral Parcels (SQLite Spatial Base)</h2>
        <div>
          <button class="btn btn-secondary" onclick="refreshParcelsTable()">🔄 Refresh</button>
          <button class="btn btn-success" onclick="downloadAllGeoJSON()">🌍 Export Full GIS GeoJSON</button>
        </div>
      </div>
      <div class="table-card">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Survey No</th>
              <th>Taluk / Village</th>
              <th>Scale</th>
              <th>Lines</th>
              <th>Vertices</th>
              <th>Timestamp</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="parcelsTableBody">
            <tr><td colspan="8" style="text-align:center; color:var(--text-muted);">No records found. Run a digitizer job to store parcels.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <script>
    let currentGeoJSON = null;

    function switchTab(tab) {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
      if (tab === 'studio') {
        document.querySelector('.nav-tabs button:nth-child(1)').classList.add('active');
        document.getElementById('tab-studio').classList.add('active');
      } else {
        document.querySelector('.nav-tabs button:nth-child(2)').classList.add('active');
        document.getElementById('tab-database').classList.add('active');
        refreshParcelsTable();
      }
    }

    async function loadSampleFMB() {
      document.getElementById('loading').style.display = 'block';
      try {
        const res = await fetch('/api/sample', { method: 'POST' });
        const data = await res.json();
        renderResults(data);
      } catch (e) {
        alert('Error executing sample pipeline: ' + e);
      } finally {
        document.getElementById('loading').style.display = 'none';
      }
    }

    async function handleFileUpload(e) {
      const file = e.target.files[0];
      if (!file) return;
      document.getElementById('loading').style.display = 'block';
      const formData = new FormData();
      formData.append('file', file);
      formData.append('survey_no', 'Upload/' + file.name.split('.')[0]);

      try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        renderResults(data);
      } catch (e) {
        alert('Error uploading and processing file: ' + e);
      } finally {
        document.getElementById('loading').style.display = 'none';
      }
    }

    function renderResults(data) {
      if (!data.success) {
        alert('Processing failed: ' + data.message);
        return;
      }
      currentGeoJSON = data.geojson;
      document.getElementById('metricsRow').style.display = 'grid';
      document.getElementById('stagesRow').style.display = 'grid';

      document.getElementById('metricSurvey').innerText = data.parcel.survey_no || '6/10A';
      document.getElementById('metricLines').innerText = data.parcel.total_lines;
      document.getElementById('metricCorners').innerText = data.parcel.total_corners;

      document.getElementById('imgBinary').src = data.previews.binary;
      document.getElementById('imgOverlay').src = data.previews.overlay;
      document.getElementById('imgOutline').src = data.previews.outline;
    }

    function downloadGeoJSON() {
      if (!currentGeoJSON) return;
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentGeoJSON, null, 2));
      const a = document.createElement('a');
      a.href = dataStr;
      a.download = "parcel_features.geojson";
      a.click();
    }

    async function refreshParcelsTable() {
      const res = await fetch('/api/parcels');
      const parcels = await res.json();
      const tbody = document.getElementById('parcelsTableBody');
      tbody.innerHTML = '';
      if (parcels.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color:var(--text-muted);">No records found in GIS DB.</td></tr>';
        return;
      }
      parcels.forEach(p => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>#${p.id}</strong></td>
          <td>${p.survey_no}</td>
          <td>${p.taluk} / ${p.village}</td>
          <td>${p.scale}</td>
          <td>${p.total_lines}</td>
          <td>${p.total_corners}</td>
          <td>${p.created_at}</td>
          <td>
            <a href="/api/parcels/${p.id}/geojson" target="_blank" class="btn btn-secondary" style="padding:0.25rem 0.6rem; font-size:0.75rem; text-decoration:none;">GeoJSON</a>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    async function downloadAllGeoJSON() {
      window.open('/api/parcels/export/geojson', '_blank');
    }
  </script>
</body>
</html>
"""


class FMBWebRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if path == "/api/parcels":
            parcels = gis_db.list_parcels()
            self._send_json(parcels)
            return

        if path == "/api/parcels/export/geojson":
            geojson = gis_db.get_all_geojson()
            self._send_json(geojson, download_name="all_fmb_parcels.geojson")
            return

        if path == "/api/parcels/export/postgis_sql":
            parcels = gis_db.list_parcels()
            schema_path = Path("schema_postgis.sql")
            full_sql = schema_path.read_text(encoding="utf-8") if schema_path.exists() else "-- PostGIS Schema\n"
            sample_path = Path("samples/user_fmb_input-1.png")
            from parcel_extractor import extract_parcel_from_source
            for p in parcels:
                rec = gis_db.get_parcel(p["id"])
                if rec:
                    p_obj, _ = extract_parcel_from_source(sample_path, survey_no=rec["survey_no"])
                    sql_insert = postgis_repo.generate_postgis_sql(p_obj, rec)
                    full_sql += f"\n\n-- Parcel #{rec['id']} ({rec['survey_no']})\n{sql_insert}"

            self._send_text(full_sql, "text/plain; charset=utf-8", download_name="postgis_cadastre_dump.sql")
            return

        if path.startswith("/api/parcels/") and path.endswith("/postgis_sql"):
            parts = path.split("/")
            parcel_id = int(parts[3])
            rec = gis_db.get_parcel(parcel_id)
            if rec:
                sample_path = Path("samples/user_fmb_input-1.png")
                from parcel_extractor import extract_parcel_from_source
                parcel, _ = extract_parcel_from_source(sample_path, survey_no=rec["survey_no"])
                sql = postgis_repo.export_sql_dump(parcel, rec)
                self._send_text(sql, "text/plain; charset=utf-8", download_name=f"parcel_{parcel_id}_postgis.sql")
            else:
                self._send_error(404, "Parcel not found")
            return

        if path.startswith("/api/parcels/") and path.endswith("/geojson"):
            parts = path.split("/")
            parcel_id = int(parts[3])
            rec = gis_db.get_parcel(parcel_id)
            if rec:
                self._send_json(rec["geojson_data"], download_name=f"parcel_{parcel_id}.geojson")
            else:
                self._send_error(404, "Parcel not found")
            return

        self._send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/sample":
            sample_path = Path("samples/user_fmb_input-1.png")
            if not sample_path.exists():
                sample_path = Path("samples/real_fmb_input-1.png")
            
            result = process_for_web(sample_path, survey_no="6/10A")
            meta = {
                "survey_no": "6/10A",
                "village": "Thalakulam [10]",
                "taluk": "Bhavani",
                "district": "Erode",
                "scale": "1:848"
            }
            parcel_id = gis_db.save_parcel(result.parcel, meta)
            
            resp = result.to_dict()
            resp["parcel_id"] = parcel_id
            resp["geojson"] = result.to_geojson()
            self._send_json(resp)
            return

        if path == "/api/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Extract binary image payload from multipart upload
            # Standard delimiter scan
            boundary = self.headers.get("Content-Type", "").split("boundary=")[-1].encode()
            parts = body.split(boundary)
            image_bytes = None
            survey_no = "Uploaded/FMB"

            for part in parts:
                if b'filename="' in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end != -1:
                        image_bytes = part[header_end + 4 : -2]
                        break

            if not image_bytes:
                image_bytes = body  # raw binary fallback

            result = process_for_web(image_bytes, survey_no=survey_no)
            parcel_id = gis_db.save_parcel(result.parcel, {
                "survey_no": survey_no,
                "village": "Custom Upload",
                "taluk": "Survey Division",
                "district": "Tamil Nadu",
                "scale": "Auto"
            })

            resp = result.to_dict()
            resp["parcel_id"] = parcel_id
            resp["geojson"] = result.to_geojson()
            self._send_json(resp)
            return

        self._send_error(404, "Not Found")

    def _send_json(self, data: Any, download_name: Optional[str] = None):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, text: str, content_type: str = "text/plain", download_name: Optional[str] = None):
        payload = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, code: int, message: str):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode("utf-8"))


def start_server(port: int = PORT):
    server = HTTPServer(("0.0.0.0", port), FMBWebRequestHandler)
    print(f"[*] FMB Web App & GIS Server running at: http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    start_server(port)

