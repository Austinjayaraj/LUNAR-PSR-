"""Opt-in real local inference smoke test, using a provided image, no dataset downloads."""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('image', type=Path)
parser.add_argument('--output', type=Path, default=ROOT / 'output/pdf/lunar-analysis-report.pdf')
args = parser.parse_args()

with tempfile.TemporaryDirectory(prefix='lunar-report-smoke-') as directory:
    os.environ['REPORT_STORAGE_DIR'] = directory
    os.environ['REPORT_AUTO_GENERATE'] = 'true'
    os.environ.pop('N8N_REPORT_WEBHOOK_URL', None)
    from fastapi.testclient import TestClient
    from api.main import app
    from pypdf import PdfReader
    with TestClient(app) as client:
        with args.image.open('rb') as image:
            response = client.post('/pipeline', files={'file': (args.image.name, image, 'image/png')})
        assert response.status_code == 200, response.text
        result = response.json()
        required = ['restoration', 'hazard_analysis', 'landing_zones', 'pipeline_telemetry', 'processing_timeline']
        assert all(k in result for k in required)
        job = result['report']
        assert job.get('report_id'), job
        query = '?access_token=' + job['access_token']
        route = '/reports/' + job['report_id']
        for _ in range(120):
            metadata = client.get(route + query).json()
            if metadata['status'] in ('READY', 'FAILED'): break
            time.sleep(.5)
        assert metadata['status'] == 'READY', metadata
        pdf = client.get(route + '/download' + query)
        html = client.get(route + '/preview' + query)
        assert pdf.status_code == html.status_code == 200
        assert pdf.content.startswith(b'%PDF')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(pdf.content)
        args.output.with_suffix('.html').write_text(html.text)
        text = '\n'.join(p.extract_text() for p in PdfReader(args.output).pages)
        assert result['mission_id'] in text
        assert '7.6287' in text and '300 held-out' in text
        assert html.text.count('data:image/png;base64,') == 5
        # Regression: image restoration, maps, telemetry and existing auxiliary routes still work.
        for endpoint in ['/health', '/version', '/model-status', '/system-status', '/evaluation/latest']:
            assert client.get(endpoint).status_code == 200, endpoint
        assert result['restoration']['output_resolution'] == [result['input_info']['width'], result['input_info']['height']]
        invalid = client.post('/pipeline', files={'file': ('invalid.png', b'invalid', 'image/png')})
        assert invalid.status_code == 400 and isinstance(invalid.json()['detail'], str)
        # Storage failures preserve an otherwise successful pipeline response.
        from unittest.mock import patch
        with patch.object(app.state.report_service, 'save_analysis', side_effect=OSError('test storage failure')):
            with args.image.open('rb') as image:
                saved_failure = client.post('/pipeline', files={'file': (args.image.name, image, 'image/png')})
        assert saved_failure.status_code == 200
        assert saved_failure.json()['restoration']['image_b64']
        assert saved_failure.json()['report']['status'] == 'FAILED'
        print(json.dumps({'analysis_id': result['mission_id'], 'report_id': job['report_id'], 'status': metadata['status'],
                          'candidate_count': len(result['landing_zones']), 'pdf_bytes': len(pdf.content),
                          'pdf_pages': len(PdfReader(args.output).pages), 'embedded_visuals': 5,
                          'output': str(args.output), 'checks': 'Real inference, automatic report, PDF/HTML download, existing routes, validation error, storage failure isolation passed'}, indent=2))
