"""Isolated synthetic fixtures exercise reports only; no fabricated production data."""
import base64
import copy
import io
import json
import os
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader
from api.report_service import ReportService
from api.report_routes import create_router


def fixture():
    image = io.BytesIO()
    Image.new('L', (64, 64), 70).save(image, format='PNG')
    b64 = base64.b64encode(image.getvalue()).decode()
    aid = str(uuid.uuid4())
    return {
        'mission_id': aid, 'timestamp': '2026-10-09T00:00:00+00:00', 'software_version': 'test-fixture',
        'input_info': {'filename': '<script>alert(1)</script>.png', 'width': 64, 'height': 64, 'channels': 1},
        'restoration': {'image_b64': b64, 'model_name': 'LunarUNet', 'method': 'test-fixture'},
        'hazard_analysis': {'danger_map_b64': b64, 'safe_region_b64': b64, 'landing_zones_overlay_b64': b64, 'hazard_pixels': 42},
        'landing_zones': [{'id': 'LZ-01', 'score': 21, 'hazard_density': .25, 'clearance': 5, 'area': 512, 'rank': 1, 'bbox': [0, 0, 32, 32]}, {'id': 'LZ-02', 'score': 10, 'rank': 2}],
        'selected_lz': {'id': 'LZ-01'},
        'pipeline_telemetry': {'restoration': {'checkpoint': '/private/test/checkpoint.pth'}, 'hazard': {'mode': 'balanced', 'engine': 'test-fixture'}, 'resource_layers': {'radar': {'available': False}}},
    }, image.getvalue()

STAGES = {k: 'COMPLETE' for k in ['image_received', 'validation', 'restoration', 'hazard', 'safe_region', 'landing_zones']}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = ReportService(self.temp.name)
        self.result, self.image = fixture()
        self.aid = self.result['mission_id']
        self.token = self.service.save_analysis(self.result, self.image, stages=STAGES)
        app = FastAPI()
        app.include_router(create_router(self.service))
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def request(self, **kwargs):
        return self.client.post('/reports/generate', json={'analysis_id': self.aid, 'analysis_token': self.token, **kwargs})

    def ready(self):
        job = self.request().json()
        self.service.generate(job['report_id'])
        return job

    def url(self, job, action=''):
        return '/reports/' + job['report_id'] + action + '?access_token=' + job['access_token']

    def test_valid_completed_analysis_pdf_and_download(self):
        job = self.ready()
        response = self.client.get(self.url(job, '/download'))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b'%PDF'))
        self.assertEqual(response.headers['content-type'], 'application/pdf')
        text = ''.join(p.extract_text() for p in PdfReader(io.BytesIO(response.content)).pages)
        self.assertIn(self.aid, text)
        self.assertIn('COMPLETED', text)
        self.assertIn('Page 1', text)
        self.assertNotIn('/private/test/', text)
        self.assertIn('not calibrated probabilities', text)
        self.assertIn('not actual elevation', text)
        self.assertIn('300 held-out', text)
        self.assertIn('7.6287', text)
        self.assertIn('Current ground-truth reference evaluation: Unavailable', text)
        self.assertGreater(sum(len(p.images) for p in PdfReader(io.BytesIO(response.content)).pages), 0)

    def test_html_escape_embedded_images_and_science(self):
        job = self.ready()
        response = self.client.get(self.url(job, '/preview'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('&lt;script&gt;', response.text)
        self.assertNotIn('<script>', response.text)
        self.assertEqual(response.text.count('data:image/png;base64,'), 5)
        self.assertIn('restored grayscale image', response.text)
        self.assertIn('derived photometric/synthetic', response.text)
        self.assertIn('Unavailable', response.text)
        self.assertIn('Water-ice', response.text)
        self.assertIn("default-src 'none'", response.headers['content-security-policy'])

    def test_invalid_id(self):
        self.assertEqual(self.request(analysis_id='../secrets').status_code, 400)

    def test_nonexistent_analysis(self):
        self.assertEqual(self.request(analysis_id=str(uuid.uuid4())).status_code, 404)

    def test_in_progress_rejected(self):
        r, _ = fixture()
        token = self.service.save_analysis(r, stages={'restoration': 'RUNNING'})
        self.assertEqual(self.request(analysis_id=r['mission_id'], analysis_token=token).status_code, 409)

    def test_partial_outputs_explicitly_unavailable(self):
        r, _ = fixture()
        r['hazard_analysis'] = {}
        token = self.service.save_analysis(r, stages={**STAGES, 'hazard': 'FAILED', 'safe_region': 'SKIPPED'})
        job = self.request(analysis_id=r['mission_id'], analysis_token=token).json()
        self.service.generate(job['report_id'])
        html = self.client.get(self.url(job, '/preview')).text
        self.assertIn('PARTIALLY COMPLETED', html)
        self.assertIn('Unavailable', html)

    def test_failed_analysis_can_have_failure_report(self):
        r = {'mission_id': str(uuid.uuid4()), 'input_info': {'filename': 'failed.png'}}
        token = self.service.save_analysis(r, stages={'validation': 'FAILED'}, error='Invalid image')
        job = self.request(analysis_id=r['mission_id'], analysis_token=token).json()
        self.service.generate(job['report_id'])
        html = self.client.get(self.url(job, '/preview')).text
        self.assertIn('Invalid image', html)
        self.assertIn('FAILED', html)
        self.assertNotIn('<td>restoration</td><td>COMPLETE</td>', html)

    def test_missing_optional_metrics(self):
        job = self.ready()
        self.assertEqual(self.client.get(self.url(job)).json()['status'], 'READY')
        html = self.client.get(self.url(job, '/preview')).text
        self.assertIn('Unavailable', html)
        self.assertNotIn('None', html)

    def test_capability_access_control(self):
        self.assertEqual(self.request(analysis_token='someone-else').status_code, 403)
        job = self.ready()
        for action in ['', '/download', '/preview']:
            self.assertEqual(self.client.get('/reports/' + job['report_id'] + action + '?access_token=someone-else').status_code, 403)
        self.assertEqual(self.client.get('/reports/internal/analyses/' + self.aid).status_code, 503)

    def test_duplicate_and_selection_aware_jobs(self):
        one = self.request().json()
        self.assertEqual(one['report_id'], self.request().json()['report_id'])
        two = self.request(selected_lz_id='LZ-02').json()
        self.assertNotEqual(one['report_id'], two['report_id'])
        self.assertEqual(self.request(selected_lz_id='LZ-INVALID').status_code, 400)
        self.service.generate(two['report_id'])
        self.assertIn('<td>selected candidate</td><td>LZ-02</td>', self.client.get(self.url(two, '/preview')).text)

    def test_concurrent_jobs_idempotent(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = list(pool.map(lambda _: self.service.create(self.aid, self.token), range(8)))
        self.assertEqual(len({j['report_id'] for j in jobs}), 1)

    def test_report_failure_keeps_analysis_and_retry_works(self):
        job = self.request().json()
        with patch('api.report_service.render_pdf', side_effect=RuntimeError('/private/password')):
            self.service.generate(job['report_id'])
        failed = self.client.get(self.url(job)).json()
        self.assertEqual(failed['status'], 'FAILED')
        self.assertNotIn('/private/password', failed['error'])
        self.assertEqual(self.service.analysis(self.aid, self.token)['status'], 'COMPLETED')
        retry = self.request(retry=True).json()
        self.assertEqual(job['report_id'], retry['report_id'])
        self.service.generate(job['report_id'])
        self.assertEqual(self.client.get(self.url(job)).json()['status'], 'READY')

    def test_not_ready_and_missing_artifact(self):
        job = self.request().json()
        self.assertEqual(self.client.get(self.url(job, '/download')).status_code, 409)
        self.service.generate(job['report_id'])
        self.service.artifact(job['report_id'], 'pdf').unlink()
        self.assertEqual(self.client.get(self.url(job, '/download')).status_code, 404)

    def test_symlink_and_traversal_blocked(self):
        job = self.ready()
        path = self.service.artifact(job['report_id'], 'pdf')
        path.unlink()
        path.symlink_to(Path(self.temp.name).parent / 'secret.pdf')
        self.assertEqual(self.client.get(self.url(job, '/download')).status_code, 400)
        with self.assertRaises(Exception): self.service.artifact('../../secret', 'pdf')

    def test_storage_initialization_is_lazy(self):
        blocked = Path(self.temp.name) / 'blocked'
        blocked.write_text('fixture file prevents directory creation')
        service = ReportService(blocked)
        with self.assertRaises(OSError):
            service.save_analysis(self.result, stages=STAGES)

    def test_render_idempotent(self):
        job = self.ready()
        with patch('api.report_service.render_pdf') as renderer:
            result = self.service.generate(job['report_id'])
        self.assertEqual(result['status'], 'READY')
        renderer.assert_not_called()

    def test_durable_restart(self):
        job = self.request().json()
        service = ReportService(self.temp.name)
        service.generate(job['report_id'])
        self.assertEqual(service.job(job['report_id'], job['access_token'])['status'], 'READY')

    def test_n8n_authenticated_contract_and_id_match(self):
        job = self.request().json()
        with patch.dict(os.environ, {'REPORT_INTERNAL_TOKEN': 'test-internal', 'N8N_REPORT_WEBHOOK_URL': 'https://fixture.invalid/webhook', 'N8N_REPORT_WEBHOOK_SECRET': 'test-webhook'}):
            with patch('api.report_service.httpx.Client') as client:
                self.service.dispatch(job['report_id'])
                call = client.return_value.__enter__.return_value.post.call_args
                self.assertEqual(call.kwargs['headers'], {'X-Lunar-Webhook-Secret': 'test-webhook'})
                payload = call.kwargs['json']
                self.assertEqual(payload['analysis_id'], self.aid)
                self.assertEqual(payload['report_id'], job['report_id'])
                self.assertNotIn('access_token', payload)
                self.assertNotIn('image_b64', payload)
            headers = {'X-Lunar-Report-Token': 'test-internal'}
            route = '/reports/internal/analyses/' + self.aid
            self.assertEqual(self.client.get(route).status_code, 401)
            record = self.client.get(route, headers=headers).json()
            self.assertEqual(record['status'], 'COMPLETED')
            wrong = self.client.post('/reports/internal/render', headers=headers, json={'analysis_id': str(uuid.uuid4()), 'report_id': job['report_id']})
            self.assertEqual(wrong.status_code, 409)
            done = self.client.post('/reports/internal/render', headers=headers, json={'analysis_id': self.aid, 'report_id': job['report_id']})
            self.assertEqual(done.json()['status'], 'READY')
            final = self.client.get('/reports/internal/jobs/' + job['report_id'], headers=headers).json()
            self.assertEqual(final['status'], 'READY')
            self.assertNotIn('access_token', final)

    def test_n8n_delivery_failure_and_timeout(self):
        job = self.request().json()
        with patch.dict(os.environ, {'REPORT_INTERNAL_TOKEN': 'fixture', 'N8N_REPORT_WEBHOOK_URL': 'https://fixture.invalid', 'N8N_REPORT_WEBHOOK_SECRET': 'fixture'}):
            with patch('api.report_service.httpx.Client', side_effect=RuntimeError('offline')):
                self.service.dispatch(job['report_id'])
            self.assertEqual(self.service.job(job['report_id'], internal=True)['status'], 'FAILED')
            self.request(retry=True)
            with self.service.connect() as db:
                db.execute("UPDATE reports SET status='AWAITING_N8N', updated_at=?", (time.time()-400,))
            self.service.tick()
            self.assertEqual(self.service.job(job['report_id'], internal=True)['status'], 'FAILED')

    def test_importable_workflow_contract(self):
        workflow = json.loads(Path('workflows/lunar-analysis-report.n8n.json').read_text())
        self.assertFalse(workflow['active'])
        nodes = {n['name']: n for n in workflow['nodes']}
        self.assertEqual(nodes['Report Webhook']['parameters']['authentication'], 'headerAuth')
        self.assertEqual(nodes['Generate Report']['parameters']['method'], 'POST')
        self.assertIn('/reports/internal/render', nodes['Generate Report']['parameters']['url'])
        self.assertIn('analysis_id', nodes['Validate Job']['parameters']['jsCode'])
        self.assertIn('report_id', nodes['Generate Report']['parameters']['jsonBody'])
        self.assertNotIn('credentials', nodes['Generate Report'])


if __name__ == '__main__': unittest.main()
