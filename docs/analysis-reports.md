# Automated analysis reports

Reports use the saved response from `POST /pipeline`, never a second inference call. FastAPI persists the analysis, uploaded image converted to a display PNG, source metadata, input statistics, stage evidence and a historical evaluation snapshot in SQLite. PDF and self-contained HTML files live on the same private persistent volume. Existing response fields remain intact; `analysis_token` and optional `report` metadata are additive.

The new ReportLab dependency composes print-friendly PDFs with images, tables and page numbering without Chromium or an external PDF service. `httpx` handles authenticated webhook delivery. SQLite uses Python's standard library; no database migration or server is required. Frontend test dependencies are development-only.

## Run locally

Use Python 3.10+ for a fresh environment and Node 22.22.2+ in the Node 22 line, or Node 24.15+ (the locked test dependencies require these versions). Run from the repository root so the existing model checkpoint path resolves. The existing checkpoint `weights/lunar_unet_v2_best.pth` is required; tests do not download datasets or weights.

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements-dev.txt
cp .env.example .env  # Only on fresh setup; preserve any existing .env and secrets
venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

In another terminal:

```sh
cd lunar-ui
npm ci
npm run dev -- --host 127.0.0.1
```

Open the URL printed by Vite. Its API proxy targets the local backend. Upload an image and run the existing site analysis. An automatic report job appears below the result map. When ready, choose **Download PDF Report** or **Preview HTML Report**. If the selected candidate changes, choose **Generate Report for Selected Candidate** to capture that selection in a separate report. The original generated overlay remains labeled as the pipeline selection.

Set `REPORT_AUTO_GENERATE=false` to use **Generate Analysis Report** on demand. Failed rendering or delivery shows a retry action; successful image analysis remains complete. A report-storage failure preserves analysis outputs but cannot be retried for that analysis because there is no persisted record: repair storage and run analysis again. Failed image-analysis responses retain a string `detail` and add an analysis ID and capability when failure recording succeeds; those records can produce a failure summary via the report API.

## Configuration

| Variable | Default / purpose |
| --- | --- |
| `REPORT_AUTO_GENERATE` | `true`; queue a job after successful analysis. |
| `REPORT_STORAGE_DIR` | `storage/reports` under the repo unless specified; use an absolute private persistent directory in deployment. |
| `REPORT_JOB_TIMEOUT_SECONDS` | `300`; mark a stalled rendering/workflow job failed, allowing an explicit retry. Must exceed expected render duration and n8n request timeout. |
| `N8N_REPORT_WEBHOOK_URL` | Empty for local generation; otherwise the complete **production** webhook URL. |
| `N8N_REPORT_WEBHOOK_SECRET` | Shared secret for inbound n8n Header Auth, header `X-Lunar-Webhook-Secret`. |
| `REPORT_INTERNAL_TOKEN` | Separate machine credential for n8n-to-FastAPI, header `X-Lunar-Report-Token`. |
| `VITE_API_BASE_URL` | Frontend build setting. Empty for a same-origin reverse proxy; otherwise the browser-reachable HTTPS API origin (no trailing path). |

Keep credentials in environment configuration and n8n's credentials store, never workflow JSON. `.env`, report storage and local dependency folders are ignored. This repo's existing `.env` loader takes precedence for any key present there, so edit that file or remove duplicate keys when using container environment variables. `PORT` and `HOST` in the old example do not configure uvicorn; use its command-line arguments.

The analysis result carries an opaque `analysis_token`; job creation requires it. A job carries a separate `access_token`, required to retrieve metadata, PDF or HTML. There are no existing user accounts; these capabilities provide access control, not user identity or login. Anyone holding a token can access its record. Don't share tokenized links. Download/preview links use query tokens so normal browser links work; disable/redact query-string access logging at uvicorn and the reverse proxy. Responses use `no-store` and `no-referrer`; HTML is escaped, images are embedded PNGs, and preview responses have a restrictive CSP. Internal endpoints never return tokens, uploaded images or private filesystem paths.

Use a private storage volume with appropriate OS permissions and backups. Files remain accessible after jobs finish and process restarts. No automatic deletion policy is enabled. For retention, archive the database and corresponding PDFs/HTML together, or delete expired records/artifacts during a maintenance window after backup. Never expose the storage directory as a static site. Old browser-only analyses from before this feature cannot be reported without rerunning them. Keep the analysis/report tokens from the returned result if access is needed after browser reset; the app currently has no report history screen.

## API contract

| Endpoint | Authentication | Behavior |
| --- | --- | --- |
| `POST /reports/generate` | JSON `analysis_token` | JSON `analysis_id`, optional `selected_lz_id`, `retry`; returns 202 with job metadata and its access token. |
| `GET /reports/{report_id}` | Query `access_token` | Job metadata/status; never returns its capability again. |
| `GET /reports/{report_id}/download` | Query `access_token` | Completed PDF attachment; 409 until READY. |
| `GET /reports/{report_id}/preview` | Query `access_token` | Completed self-contained HTML. |
| `GET /reports/internal/analyses/{analysis_id}` | Header `X-Lunar-Report-Token` | Stage evidence, timestamp, count and backend result reference; no large binaries. |
| `POST /reports/internal/render` | Same machine header | JSON `analysis_id`, `report_id`; validates correspondence and renders the saved record. |
| `GET /reports/internal/jobs/{report_id}` | Same machine header | Final durable status and artifact references for n8n. |

IDs must be canonical UUIDs. Unknown IDs return 404, invalid IDs 400, wrong capabilities 403, invalid machine credentials 401, unconfigured orchestration 503, unfinished analysis 409. Public routes require capabilities regardless of CORS. Files are server-selected UUID paths, checked against the configured directory; symlinks/path traversal are rejected.

Job statuses: `PENDING`, `AWAITING_N8N`, `RUNNING`, `READY`, `FAILED`. Analysis statuses are independent: `COMPLETED` means all required stages and outputs exist; `PARTIALLY COMPLETED` means usable outputs exist but some stages/required outputs are missing or failed; `FAILED` means no usable restoration/hazard output; `IN_PROGRESS` records cannot yet be reported. Missing optional values do not turn a complete analysis into a failure.

The unique `(analysis_id, selected_candidate)` constraint avoids duplicate jobs. Retries reuse a failed job's ID and capability; ready reports remain stable. Atomic claims protect duplicate render calls. A background worker scans the durable queue every two seconds, handles delivery/rendering failures, and marks interrupted jobs failed after the configured timeout. This intentionally fits a single-backend-instance demonstration deployment; for multiple replicas or long-running jobs, use a shared queue/storage service and explicit leases rather than placing SQLite on network filesystems.

## Import and configure n8n

1. Import `workflows/lunar-analysis-report.n8n.json` using n8n's workflow import menu. It is inactive and contains no credentials. It does not run inference or manipulate image binaries.
2. Create a **Header Auth** credential for the Webhook node: name `X-Lunar-Webhook-Secret`, value matching `N8N_REPORT_WEBHOOK_SECRET`. Select it on **Report Webhook**.
3. Create a separate **Header Auth** credential for HTTP Request nodes: name `X-Lunar-Report-Token`, value matching `REPORT_INTERNAL_TOKEN`. Select it on **Retrieve Analysis**, **Generate Report** and **Record Final Status**.
4. Replace `https://YOUR-BACKEND-ORIGIN.example` in those three nodes with the fixed backend origin reachable **from n8n**. Keep the existing ID expressions and endpoint paths. The workflow deliberately doesn't accept callback URLs from webhook payloads.
5. Activate the workflow and copy its production webhook URL into `N8N_REPORT_WEBHOOK_URL`; restart FastAPI. Use the test webhook only while n8n is explicitly listening for a test event.
6. Restrict the webhook and internal API routes to trusted network sources using reverse-proxy/firewall rules, and use TLS whenever crossing networks. Configure the Webhook's available IP restrictions for the FastAPI source as appropriate.

Flow: authenticated webhook -> validate event/schema/UUIDs -> retrieve saved analysis metadata -> validate analysis identity/completion -> ask FastAPI to render -> retrieve final durable status/reference -> fail the workflow if not READY. FastAPI commits final status before returning render results; n8n records that reference in the final execution node. An immediate webhook acknowledgment only means delivery was accepted. If n8n later fails before rendering, FastAPI times out the job and the frontend offers retry. Rendering errors are persisted immediately as FAILED and surfaced in both systems.

Webhook payload (IDs below are placeholders, use values from a real returned job):

```json
{
  "event": "analysis.report.requested",
  "schema_version": 1,
  "analysis_id": "<canonical analysis UUID>",
  "report_id": "<canonical report UUID>"
}
```

The HTTP Request render body is `{ "analysis_id": "...", "report_id": "..." }`. The workflow uses HTTP Request v4.2, Webhook v2 and Code v2 nodes. Confirm those nodes and credentials after import in your installed n8n version. Generate Report has a 240-second request timeout; set the report job timeout above it if increasing this value.

## Deployment addressing

The frontend's URL and n8n's backend URL can differ. A browser needs a publicly/reliably reachable API origin or a same-origin reverse proxy. In Docker, `127.0.0.1` addresses the current container: use a Compose service hostname (for example `http://backend:8000`) between n8n and FastAPI on a private network. On Docker Desktop, `host.docker.internal` can reach a host backend only if it listens on a reachable interface. n8n Cloud needs a reachable HTTPS endpoint; a local loopback URL will not work. Mount `REPORT_STORAGE_DIR` persistently so reports survive container replacement. For same-origin production, forward the API paths configured in `lunar-ui/vite.config.js`, including `/reports`, to FastAPI. Frontend configuration is baked in by `npm run build`; Vite's dev proxy is not included in the production build.

## Scientific content

Current results include actual input/source metadata, restored-image diagnostics, model/checkpoint telemetry, hazard pixel fractions, algorithmic masks, candidate scores and backend timings. The report does not infer source geography, physical resolution, elevation or per-image reference metrics. Historical restoration metrics come from `evaluation_results/lunar_unet_complete_metrics.json` and are explicitly labeled as a separate 300-image held-out evaluation. Historical hazard metrics come from `evaluation_results/v2_final/v2_final_comparison.json`, selected by actual hazard mode and labeled against a derived photometric/synthetic reference. The pipeline's legacy default Canny/dilation parameters are not reported as the V2 active configuration. Missing outputs are Unavailable; unavailable water-ice/thermal/spectral/radar layers remain explicit. Scores and safe-region masks are not certified landing safety.

## Verification

```sh
venv/bin/python -m unittest discover -s tests -v
cd lunar-ui
npm test
npm run build
```

Optional real local checkpoint/inference smoke test (uses an existing image, downloads nothing):

```sh
venv/bin/python scripts/verify_report_pipeline.py data/processed_pairs/lr/15.png
pdftoppm -png output/pdf/lunar-analysis-report.pdf tmp/pdfs/report
```

The script performs real inference, waits for automatic generation, downloads PDF/HTML, checks five embedded visuals and historical labels, checks existing API routes, rejects invalid images, and simulates report-storage failure to confirm successful analysis is preserved. It uses temporary report storage and exports the sample report for inspection.

To verify live n8n: configure and activate the imported workflow, run a new real analysis, inspect the execution's matching analysis/report IDs and final READY status, download the PDF/HTML from the frontend, restart FastAPI and recheck the same tokenized artifact URL. Then temporarily stop n8n and verify delivery failure/retry; restore it and retry the same job. Confirm no second `/pipeline` request occurs during reporting.

Live n8n has **not** been tested in this environment. Automated tests cover mocked delivery, authenticated metadata/render/status contracts, matching IDs, failures/timeouts and workflow structure. Backend tests also cover PDF downloads, escaping, partial/failed analysis, unavailable fields/resources, historical labels, limitations, durable restarts, candidate selections, duplicates/concurrent requests and capability/path security. Frontend tests cover auto polling, downloads, retries, selection changes and stale-response rejection. Browser verification also covers real upload, automatic readiness, frontend PDF download, and entering the existing terrain view. Recording was not exercised interactively; its components were not changed.


## Implemented file inventory and observed results

Created: `api/report_service.py`, `api/report_rendering.py`, `api/report_routes.py`, `tests/test_reports.py`, `scripts/verify_report_pipeline.py`, `workflows/lunar-analysis-report.n8n.json`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `lunar-ui/.env.example`, `lunar-ui/src/components/AnalysisReport.jsx`, `lunar-ui/src/components/AnalysisReport.test.jsx`, this guide, and `docs/report-implementation-plan.md`.

Modified: `api/main.py`, `api/schemas.py`, `.env.example`, `lunar-ui/src/api.js`, `lunar-ui/src/store.js`, `lunar-ui/src/components/AnalysisViewport.jsx`, `lunar-ui/src/index.css`, `lunar-ui/vite.config.js`, `lunar-ui/package.json`, and `lunar-ui/package-lock.json`. No Git repository exists in this checkout, so no commit or Git diff was created.

Generated inspection artifacts: `output/pdf/lunar-analysis-report.pdf`, its self-contained `.html` preview, and `output/pdf/report-ui.jpg`. The sample used real local `data/processed_pairs/lr/15.png` with the existing checkpoint: two candidates, five embedded visuals, nine PDF pages. Input images and maps in test fixtures are explicitly synthetic; the exported sample and browser run use actual pipeline outputs.

Observed checks: 29 backend tests passed (including existing evaluation tests); six frontend tests passed; scoped ESLint for the report component, tests and API client passed; Vite production build passed with an existing large-bundle advisory; npm audit reported zero vulnerabilities after compatible fixes. Real inference smoke checks and browser upload/download/terrain transition passed. All nine PDF pages were rendered and visually reviewed after fixing caption pagination. No claim is made that live n8n or recording was tested.
