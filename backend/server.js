/**
 * Node.js REST API Server for FMB Digitizer & PostGIS Cadastre Hub.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = process.env.PORT || 3000;
const ROOT_DIR = path.resolve(__dirname, '..');
const UPLOAD_DIR = path.join(ROOT_DIR, 'uploads');
const OUTPUT_DIR = path.join(ROOT_DIR, 'outputs');

if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

function executePythonDigitizer(filePath, surveyNo) {
  return new Promise((resolve, reject) => {
    const pythonScript = `
import json
import sys
from pathlib import Path
from backend.pipeline import process_for_web
from database.postgis_repository import PostGISRepository
from database.sqlite_repository import SQLiteGISRepository

file_path = sys.argv[1]
survey_no = sys.argv[2] if len(sys.argv) > 2 else "6/10A"

result = process_for_web(file_path, survey_no=survey_no)
meta = {
    "survey_no": survey_no,
    "village": "Thalakulam [10]",
    "taluk": "Bhavani",
    "district": "Erode",
    "scale": "1:848"
}

db = SQLiteGISRepository("fmb_gis.db")
parcel_id = db.save_parcel(result.parcel, meta)

repo = PostGISRepository()
postgis_sql = repo.generate_postgis_sql(result.parcel, meta)

resp = result.to_dict()
resp["parcel_id"] = parcel_id
resp["geojson"] = result.to_geojson()
resp["postgis_sql"] = postgis_sql

print(json.dumps(resp))
`;

    const pyProcess = spawn('./venv/bin/python', ['-c', pythonScript, filePath, surveyNo], {
      cwd: ROOT_DIR
    });

    let stdout = '';
    let stderr = '';

    pyProcess.stdout.on('data', (data) => { stdout += data.toString(); });
    pyProcess.stderr.on('data', (data) => { stderr += data.toString(); });

    pyProcess.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`Python pipeline failed (code ${code}): ${stderr}`));
      }
      try {
        const jsonStartIndex = stdout.indexOf('{');
        if (jsonStartIndex === -1) throw new Error('No JSON output found');
        const jsonResult = JSON.parse(stdout.substring(jsonStartIndex));
        resolve(jsonResult);
      } catch (err) {
        reject(new Error(`Failed to parse Python JSON output: ${err.message}\nRaw: ${stdout}`));
      }
    });
  });
}

function fetchParcelsFromDB() {
  return new Promise((resolve, reject) => {
    const script = `
import json
from database.sqlite_repository import SQLiteGISRepository
db = SQLiteGISRepository("fmb_gis.db")
print(json.dumps(db.list_parcels()))
`;
    const py = spawn('./venv/bin/python', ['-c', script], { cwd: ROOT_DIR });
    let stdout = '';
    py.stdout.on('data', (d) => { stdout += d.toString(); });
    py.on('close', () => {
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch (e) {
        resolve([]);
      }
    });
  });
}

function fetchParcelGeoJSON(parcelId) {
  return new Promise((resolve, reject) => {
    const script = `
import json
import sys
from database.sqlite_repository import SQLiteGISRepository
db = SQLiteGISRepository("fmb_gis.db")
p = db.get_parcel(int(sys.argv[1]))
print(json.dumps(p["geojson_data"] if p else {}))
`;
    const py = spawn('./venv/bin/python', ['-c', script, String(parcelId)], { cwd: ROOT_DIR });
    let stdout = '';
    py.stdout.on('data', (d) => { stdout += d.toString(); });
    py.on('close', () => {
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch (e) {
        resolve(null);
      }
    });
  });
}

const server = http.createServer(async (req, res) => {
  const parsedUrl = new URL(req.url, `http://${req.headers.host}`);
  const pathname = parsedUrl.pathname;

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  // 1. Serve React Frontend
  if (req.method === 'GET' && (pathname === '/' || pathname === '/index.html')) {
    const htmlPath = path.join(ROOT_DIR, 'frontend', 'index.html');
    fs.readFile(htmlPath, 'utf-8', (err, content) => {
      if (err) {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        return res.end('Error loading frontend UI: ' + err.message);
      }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(content);
    });
    return;
  }

  // 2. GET /api/parcels
  if (req.method === 'GET' && pathname === '/api/parcels') {
    try {
      const parcels = await fetchParcelsFromDB();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(parcels));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ error: e.message }));
    }
  }

  // 3. GET /api/parcels/:id/geojson
  if (req.method === 'GET' && pathname.startsWith('/api/parcels/') && pathname.endsWith('/geojson')) {
    const parts = pathname.split('/');
    const parcelId = parseInt(parts[3], 10);
    const geojson = await fetchParcelGeoJSON(parcelId);
    if (!geojson) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ error: 'Parcel not found' }));
    }
    res.writeHead(200, {
      'Content-Type': 'application/json',
      'Content-Disposition': `attachment; filename="parcel_${parcelId}.geojson"`
    });
    return res.end(JSON.stringify(geojson, null, 2));
  }

  // 4. POST /api/digitize (Live File Upload)
  if (req.method === 'POST' && pathname === '/api/digitize') {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', async () => {
      try {
        const bodyBuffer = Buffer.concat(chunks);
        const contentType = req.headers['content-type'] || '';
        let fileBuffer = bodyBuffer;
        let filename = `fmb_${Date.now()}.png`;
        let surveyNo = '6/10A';

        if (contentType.includes('boundary=')) {
          const boundary = contentType.split('boundary=')[1];
          const parts = bodyBuffer.toString('binary').split(`--${boundary}`);
          for (const part of parts) {
            if (part.includes('filename="')) {
              const filenameMatch = part.match(/filename="(.+?)"/);
              if (filenameMatch) filename = filenameMatch[1];
              const headerEnd = part.indexOf('\r\n\r\n');
              if (headerEnd !== -1) {
                const fileBinary = part.substring(headerEnd + 4, part.length - 2);
                fileBuffer = Buffer.from(fileBinary, 'binary');
              }
            }
            if (part.includes('name="survey_no"')) {
              const headerEnd = part.indexOf('\r\n\r\n');
              if (headerEnd !== -1) {
                surveyNo = part.substring(headerEnd + 4, part.length - 2).trim();
              }
            }
          }
        }

        const uploadFilePath = path.join(UPLOAD_DIR, filename);
        fs.writeFileSync(uploadFilePath, fileBuffer);

        let processFilePath = uploadFilePath;
        if (filename.toLowerCase().endsWith('.pdf')) {
          const pngStem = path.join(UPLOAD_DIR, `pdf_converted_${Date.now()}`);
          await new Promise((resolve, reject) => {
            const ppm = spawn('pdftoppm', ['-png', '-r', '300', uploadFilePath, pngStem]);
            ppm.on('close', (code) => {
              if (code === 0 && fs.existsSync(`${pngStem}-1.png`)) {
                processFilePath = `${pngStem}-1.png`;
                resolve();
              } else {
                reject(new Error('Failed to convert uploaded PDF to image'));
              }
            });
          });
        }

        const result = await executePythonDigitizer(processFilePath, surveyNo);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, message: err.message }));
      }
    });
    return;
  }

  // 5. POST /api/sample
  if (req.method === 'POST' && pathname === '/api/sample') {
    try {
      let samplePath = path.join(ROOT_DIR, 'samples', 'user_fmb_input-1.png');
      if (!fs.existsSync(samplePath)) {
        samplePath = path.join(ROOT_DIR, 'samples', 'real_fmb_input-1.png');
      }
      const result = await executePythonDigitizer(samplePath, '6/10A');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: false, message: err.message }));
    }
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Endpoint Not Found' }));
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`[🚀 Node.js + React FMB Digitizer Server] Running at: http://0.0.0.0:${PORT}`);
});

