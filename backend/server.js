/**
 * FMB Cadastral Digitizer — Node.js REST API Server
 *
 * Routes:
 *   GET  /                        → React SPA
 *   POST /api/digitize            → Upload FMB (image/PDF) → Full M1-M5 pipeline
 *   POST /api/sample              → Run pipeline on bundled sample
 *   GET  /api/parcels             → List all stored parcels
 *   GET  /api/parcels/:id         → Get parcel details
 *   GET  /api/parcels/:id/geojson → Download GeoJSON
 *   GET  /api/status              → DB backend status
 */

const http = require('http');
const fs   = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT      = process.env.PORT || 3000;
const ROOT_DIR  = path.resolve(__dirname, '..');
const UPLOAD_DIR = path.join(ROOT_DIR, 'uploads');
const OUTPUT_DIR = path.join(ROOT_DIR, 'outputs');

if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// ---------------------------------------------------------------------------
// Python bridge helpers
// ---------------------------------------------------------------------------

function runPython(scriptLines, args = []) {
  return new Promise((resolve, reject) => {
    const script = scriptLines.join('\n');
    const py = spawn('./venv/bin/python', ['-c', script, ...args], {
      cwd: ROOT_DIR,
      env: { ...process.env }
    });

    let stdout = '', stderr = '';
    py.stdout.on('data', d => { stdout += d.toString(); });
    py.stderr.on('data', d => { stderr += d.toString(); });

    py.on('close', code => {
      if (code !== 0) return reject(new Error(`Python error (code ${code}): ${stderr.slice(0, 400)}`));
      const start = stdout.indexOf('{');
      if (start === -1) return reject(new Error('No JSON from Python. stderr: ' + stderr.slice(0, 400)));
      try {
        resolve(JSON.parse(stdout.substring(start)));
      } catch (e) {
        reject(new Error('JSON parse failed: ' + e.message));
      }
    });
  });
}

function digitize(filePath, surveyNo = '6/10A') {
  return runPython([
    'import json, sys',
    'from backend.pipeline import process_for_web',
    'from database.repository import GISRepository',
    'file_path = sys.argv[1]',
    'survey_no = sys.argv[2] if len(sys.argv) > 2 else "6/10A"',
    'result = process_for_web(file_path, survey_no=survey_no)',
    'meta = {',
    '  "survey_no": survey_no,',
    '  "village": "Thalakulam [10]",',
    '  "taluk": "Bhavani",',
    '  "district": "Erode",',
    '  "scale": "1:848",',
    '  "scale_denominator": result.parcel.scale_ratio or 848',
    '}',
    'repo = GISRepository()',
    'parcel_id = repo.save_parcel(result.parcel, meta, result.cadastral_polygon)',
    'd = result.to_dict()',
    'd["parcel_id"] = parcel_id',
    'd["db_backend"] = repo.backend',
    'd["geojson"] = result.to_geojson()',
    'if result.cadastral_polygon:',
    '  d["polygon"] = result.cadastral_polygon.to_dict()',
    'print(json.dumps(d))',
  ], [filePath, surveyNo]);
}

function listParcels() {
  return runPython([
    'import json',
    'from database.repository import GISRepository',
    'repo = GISRepository()',
    'd = repo.list_parcels()',
    'print(json.dumps({"parcels": d, "backend": repo.backend}))',
  ]);
}

function getParcel(parcelId) {
  return runPython([
    'import json, sys',
    'from database.repository import GISRepository',
    'repo = GISRepository()',
    'p = repo.get_parcel(int(sys.argv[1]))',
    'print(json.dumps(p or {}))',
  ], [String(parcelId)]);
}

function getStatus() {
  return runPython([
    'import json',
    'from database.connection import DBConnection',
    'be = DBConnection.backend_name()',
    'pg = DBConnection.postgres_available()',
    'print(json.dumps({"backend": be, "postgresql": pg, "sqlite": True}))',
  ]);
}

function runSurvey(surveyData) {
  return runPython([
    'import json, sys',
    'from backend.survey_service import process_ground_survey',
    'data = json.loads(sys.argv[1])',
    'res = process_ground_survey(',
    '  parcel_id=int(data.get("parcel_id", 1)),',
    '  points=data.get("points", []),',
    '  anchor_fmb_indices=tuple(data.get("anchor_fmb_indices", [0, 1])),',
    '  anchor_survey_indices=tuple(data.get("anchor_survey_indices", [0, 1])),',
    '  surveyor_name=data.get("surveyor_name", "Field Surveyor"),',
    '  survey_date=data.get("survey_date", "2026-09-20")',
    ')',
    'print(json.dumps(res))',
  ], [JSON.stringify(surveyData)]);
}

function listSurveys(parcelId) {
  return runPython([
    'import json, sys',
    'from database.repository import GISRepository',
    'repo = GISRepository()',
    'pid = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "" and sys.argv[1] != "undefined" else None',
    'surveys = repo.list_field_surveys(pid)',
    'print(json.dumps({"surveys": surveys, "backend": repo.backend}))',
  ], [String(parcelId || '')]);
}

function getComparison(parcelId) {
  return runPython([
    'import json, sys',
    'from database.repository import GISRepository',
    'repo = GISRepository()',
    'pid = int(sys.argv[1])',
    'comp = repo.get_latest_comparison(pid)',
    'print(json.dumps(comp or {}))',
  ], [String(parcelId)]);
}

// ---------------------------------------------------------------------------
// Multipart file parser (no external dependencies)
// ---------------------------------------------------------------------------

function parseMultipart(buffer, contentType) {
  const boundaryMatch = contentType.match(/boundary=(.+)/);
  if (!boundaryMatch) return { fileBuffer: buffer, filename: 'upload.png', surveyNo: '6/10A' };

  const boundary = '--' + boundaryMatch[1].trim();
  const bodyStr = buffer.toString('binary');
  const parts = bodyStr.split(boundary).slice(1, -1);

  let fileBuffer = null, filename = `fmb_${Date.now()}.png`, surveyNo = '6/10A';

  for (const part of parts) {
    const sepIdx = part.indexOf('\r\n\r\n');
    if (sepIdx === -1) continue;
    const headers = part.slice(0, sepIdx);
    const body    = part.slice(sepIdx + 4, part.length - 2);

    if (headers.includes('filename=')) {
      const m = headers.match(/filename="(.+?)"/);
      if (m) filename = m[1];
      fileBuffer = Buffer.from(body, 'binary');
    } else if (headers.includes('name="survey_no"')) {
      surveyNo = body.trim();
    }
  }
  return { fileBuffer: fileBuffer || buffer, filename, surveyNo };
}

// ---------------------------------------------------------------------------
// HTTP Server
// ---------------------------------------------------------------------------

const server = http.createServer(async (req, res) => {
  const url      = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = url.pathname;
  const method   = req.method;

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (method === 'OPTIONS') { res.writeHead(204); return res.end(); }

  const json = (data, code = 200) => {
    res.writeHead(code, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
  };

  // ── Static assets ────────────────────────────────────────────────────────
  if (method === 'GET' && (pathname === '/' || pathname === '/index.html')) {
    const html = path.join(ROOT_DIR, 'frontend', 'index.html');
    return fs.readFile(html, 'utf-8', (err, content) => {
      if (err) { res.writeHead(500); return res.end(err.message); }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(content);
    });
  }

  // Serve static files from frontend/
  if (method === 'GET' && pathname.startsWith('/static/')) {
    const filePath = path.join(ROOT_DIR, 'frontend', pathname.replace('/static/', ''));
    return fs.readFile(filePath, (err, data) => {
      if (err) { res.writeHead(404); return res.end('Not found'); }
      res.writeHead(200);
      res.end(data);
    });
  }

  // ── API: Status ──────────────────────────────────────────────────────────
  if (method === 'GET' && pathname === '/api/status') {
    try { json(await getStatus()); }
    catch (e) { json({ error: e.message }, 500); }
    return;
  }

  // ── API: List parcels ────────────────────────────────────────────────────
  if (method === 'GET' && pathname === '/api/parcels') {
    try {
      const r = await listParcels();
      json(r);
    } catch (e) { json({ error: e.message, parcels: [] }, 500); }
    return;
  }

  // ── API: Parcel details ──────────────────────────────────────────────────
  if (method === 'GET' && pathname.match(/^\/api\/parcels\/\d+$/)) {
    const id = parseInt(pathname.split('/')[3]);
    try { json(await getParcel(id)); }
    catch (e) { json({ error: e.message }, 500); }
    return;
  }

  // ── API: GeoJSON download ────────────────────────────────────────────────
  if (method === 'GET' && pathname.match(/^\/api\/parcels\/\d+\/geojson$/)) {
    const id = parseInt(pathname.split('/')[3]);
    try {
      const p = await getParcel(id);
      const geojson = p?.geojson_data || {};
      res.writeHead(200, {
        'Content-Type': 'application/geo+json',
        'Content-Disposition': `attachment; filename="parcel_${id}.geojson"`,
      });
      res.end(JSON.stringify(geojson, null, 2));
    } catch (e) { json({ error: e.message }, 500); }
    return;
  }

  // ── API: Upload & digitize ───────────────────────────────────────────────
  if (method === 'POST' && pathname === '/api/digitize') {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', async () => {
      try {
        const body = Buffer.concat(chunks);
        const ct   = req.headers['content-type'] || '';
        const { fileBuffer, filename, surveyNo } = parseMultipart(body, ct);

        const uploadPath = path.join(UPLOAD_DIR, filename);
        fs.writeFileSync(uploadPath, fileBuffer);

        let processPath = uploadPath;
        if (filename.toLowerCase().endsWith('.pdf')) {
          const stem = path.join(UPLOAD_DIR, `pdf_${Date.now()}`);
          await new Promise((res, rej) => {
            const p = spawn('pdftoppm', ['-png', '-r', '300', uploadPath, stem]);
            p.on('close', code => {
              const pg1 = `${stem}-1.png`;
              if (code === 0 && fs.existsSync(pg1)) { processPath = pg1; res(); }
              else rej(new Error('PDF conversion failed'));
            });
          });
        }

        const result = await digitize(processPath, surveyNo);
        json(result);
      } catch (e) {
        json({ success: false, message: e.message }, 500);
      }
    });
    return;
  }

  // ── API: Run sample ──────────────────────────────────────────────────────
  if (method === 'POST' && pathname === '/api/sample') {
    try {
      let samplePath = path.join(ROOT_DIR, 'samples', 'user_fmb_input-1.png');
      if (!fs.existsSync(samplePath)) samplePath = path.join(ROOT_DIR, 'samples', 'real_fmb_input-1.png');
      if (!fs.existsSync(samplePath)) return json({ success: false, message: 'No sample image found in samples/' }, 404);
      const result = await digitize(samplePath, '6/10A');
      json(result);
    } catch (e) { json({ success: false, message: e.message }, 500); }
    return;
  }

  // ── API: Ingest Ground Survey & Run Spatial Comparison ───────────────────
  if (method === 'POST' && pathname === '/api/survey') {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', async () => {
      try {
        const bodyStr = Buffer.concat(chunks).toString('utf-8');
        const data = JSON.parse(bodyStr);
        const result = await runSurvey(data);
        json(result);
      } catch (e) {
        json({ success: false, message: e.message }, 500);
      }
    });
    return;
  }

  // ── API: List Field Surveys ──────────────────────────────────────────────
  if (method === 'GET' && pathname === '/api/surveys') {
    const parcelId = url.searchParams.get('parcel_id');
    try {
      const r = await listSurveys(parcelId);
      json(r);
    } catch (e) { json({ error: e.message, surveys: [] }, 500); }
    return;
  }

  // ── API: Latest Spatial Comparison for Parcel ────────────────────────────
  if (method === 'GET' && pathname.match(/^\/api\/comparisons\/\d+$/)) {
    const parcelId = parseInt(pathname.split('/')[3]);
    try {
      const comp = await getComparison(parcelId);
      json(comp);
    } catch (e) { json({ error: e.message }, 500); }
    return;
  }

  json({ error: 'Not found' }, 404);
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`\n╔════════════════════════════════════════════════════╗`);
  console.log(`║  FMB Cadastral Digitizer  — React + Node.js        ║`);
  console.log(`║  Server : http://localhost:${PORT}                    ║`);
  console.log(`╚════════════════════════════════════════════════════╝\n`);
});
