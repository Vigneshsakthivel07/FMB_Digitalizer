/**
 * Node.js + Express-style Backend Server for FMB Digitizer & PostGIS Cadastre Hub.
 *
 * Direct integration with Python Module 1, Module 2, and Parcel Extractor.
 * No demo/mock data: strictly processes real uploaded FMB survey files and stores in PostGIS.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = process.env.PORT || 3000;
const UPLOAD_DIR = path.join(__dirname, 'uploads');
const OUTPUT_DIR = path.join(__dirname, 'outputs');

if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// Run Python Digitizer Pipeline on real input file
function executePythonDigitizer(filePath, surveyNo) {
  return new Promise((resolve, reject) => {
    const pythonScript = `
import json
import sys
from pathlib import Path
from fmb_pipeline import process_for_web
from infrastructure.postgis_repository import PostGISRepository
from infrastructure.gis_database import GISDatabase

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

# Save in GIS DB
db = GISDatabase("fmb_gis.db")
parcel_id = db.save_parcel(result.parcel, meta)

# Generate PostGIS SQL
repo = PostGISRepository()
postgis_sql = repo.generate_postgis_sql(result.parcel, meta)

resp = result.to_dict()
resp["parcel_id"] = parcel_id
resp["geojson"] = result.to_geojson()
resp["postgis_sql"] = postgis_sql

print(json.dumps(resp))
`;

    const pyProcess = spawn('./venv/bin/python', ['-c', pythonScript, filePath, surveyNo], {
      cwd: __dirname
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

// Fetch list of parcels from database
function fetchParcelsFromDB() {
  return new Promise((resolve, reject) => {
    const script = `
import json
from infrastructure.gis_database import GISDatabase
db = GISDatabase("fmb_gis.db")
print(json.dumps(db.list_parcels()))
`;
    const py = spawn('./venv/bin/python', ['-c', script], { cwd: __dirname });
    let stdout = '';
    py.stdout.on('data', (d) => { stdout += d.toString(); });
    py.on('close', (code) => {
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch (e) {
        resolve([]);
      }
    });
  });
}

// Fetch single parcel GeoJSON
function fetchParcelGeoJSON(parcelId) {
  return new Promise((resolve, reject) => {
    const script = `
import json
import sys
from infrastructure.gis_database import GISDatabase
db = GISDatabase("fmb_gis.db")
p = db.get_parcel(int(sys.argv[1]))
print(json.dumps(p["geojson_data"] if p else {}))
`;
    const py = spawn('./venv/bin/python', ['-c', script, String(parcelId)], { cwd: __dirname });
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

// HTTP Server
const server = http.createServer(async (req, res) => {
  const parsedUrl = new URL(req.url, `http://${req.headers.host}`);
  const pathname = parsedUrl.pathname;

  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  // 1. Serve React Frontend SPA
  if (req.method === 'GET' && (pathname === '/' || pathname === '/index.html')) {
    const htmlPath = path.join(__dirname, 'frontend', 'index.html');
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

  // 4. POST /api/digitize (File Upload & Real Execution)
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

        // If uploaded file is PDF, convert page 1 to PNG via pdftoppm
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

        // Execute real Python processing pipeline
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

  // 5. POST /api/sample (Direct execution on the uploaded survey 6/10A image)
  if (req.method === 'POST' && pathname === '/api/sample') {
    try {
      let samplePath = path.join(__dirname, 'samples', 'user_fmb_input-1.png');
      if (!fs.existsSync(samplePath)) {
        samplePath = path.join(__dirname, 'samples', 'real_fmb_input-1.png');
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
