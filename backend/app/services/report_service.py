"""Durable, selection-aware report jobs backed by stdlib SQLite."""
import base64
import io
import json
import os
import secrets
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

import httpx
from PIL import Image
from fastapi import HTTPException
try:
    from backend.app.services.report_rendering import render_html, render_pdf
except ImportError:
    from api.report_rendering import render_html, render_pdf

def _find_root():
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / 'evaluation_results').exists() and (p / 'weights').exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[3]
ROOT = _find_root()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def valid_id(value):
    try:
        if str(uuid.UUID(value)) != value: raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, "Invalid analysis or report ID")
    return value


class ReportService:
    def __init__(self, directory=None):
        self.directory = Path(directory or os.getenv("REPORT_STORAGE_DIR", str(ROOT / "storage" / "reports"))).resolve()
        self.db = self.directory / "reports.sqlite3"

    @contextmanager
    def connect(self):
        # Lazy initialization keeps unavailable report storage from preventing ML startup.
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        db.executescript("""
            CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY, token TEXT NOT NULL, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reports(
                report_id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL, selection_key TEXT NOT NULL,
                access_token TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
                created_at TEXT NOT NULL, updated_at REAL NOT NULL, generated_at TEXT, selected_lz_id TEXT,
                UNIQUE(analysis_id, selection_key));
        """)
        self.db.chmod(0o600)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_analysis(self, result, image_bytes=None, input_stats=None, stages=None, error=None):
        aid = valid_id(result["mission_id"])
        token = secrets.token_urlsafe(32)
        original = None
        source = {"filename": (result.get("input_info") or {}).get("filename")}
        if image_bytes:
            with Image.open(io.BytesIO(image_bytes)) as im:
                source.update(width=im.width, height=im.height, channels=len(im.getbands()), format=im.format)
                png = io.BytesIO()
                im.convert("RGB" if len(im.getbands()) > 1 else "L").save(png, format="PNG")
                original = base64.b64encode(png.getvalue()).decode()
        stages = stages or {}
        required = ["image_received", "validation", "restoration", "hazard", "safe_region", "landing_zones"]
        usable = bool((result.get("restoration") or {}).get("image_b64") or (result.get("hazard_analysis") or {}).get("danger_map_b64"))
        complete = all(stages.get(k) == "COMPLETE" for k in required)
        ha = result.get("hazard_analysis") or {}
        complete = complete and not error and bool((result.get("restoration") or {}).get("image_b64")) and bool(ha.get("danger_map_b64")) and bool(ha.get("safe_region_b64")) and "landing_zones" in result
        running = any(v in ("RUNNING", "PENDING") for v in stages.values())
        status = "IN_PROGRESS" if running else "COMPLETED" if complete else "PARTIALLY COMPLETED" if usable else "FAILED"
        historical = {}
        rp = ROOT / "evaluation_results" / "lunar_unet_complete_metrics.json"
        if rp.exists():
            data = json.loads(rp.read_text())["primary_restoration_metrics"]
            historical["restoration"] = {"MAE (DN)": data.get("MAE"), "RMSE (DN)": data.get("RMSE"), "R2": data.get("R2"), "PSNR (dB)": data.get("PSNR_dB"), "SSIM": data.get("SSIM")}
        hp = ROOT / "evaluation_results" / "v2_final" / "v2_final_comparison.json"
        if hp.exists():
            data = json.loads(hp.read_text())["hazard_detection"]
            historical["hazard"] = {m: {k: data.get("v2_" + m, {}).get(k) for k in ["precision", "recall", "f1", "iou"]} for m in ["balanced", "precision", "conservative"]}
        record = dict(result=result, status=status, stages=stages, error=error, original_b64=original,
                      source_info=source, input_stats=input_stats, historical=historical)
        with self.connect() as db:
            db.execute("INSERT INTO analyses VALUES (?, ?, ?)", (aid, token, json.dumps(record)))
        return token

    def analysis(self, aid, token=None, internal=False):
        valid_id(aid)
        with self.connect() as db: row = db.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
        if not row: raise HTTPException(404, "Analysis not found")
        if not internal and not secrets.compare_digest(row["token"], token or ""):
            raise HTTPException(403, "Analysis access denied")
        return json.loads(row["record"])

    def create(self, aid, token, selected_lz_id=None, retry=False):
        record = self.analysis(aid, token)
        if record["status"] == "IN_PROGRESS": raise HTTPException(409, "Analysis is still in progress")
        result = record["result"]
        if selected_lz_id is None: selected_lz_id = (result.get("selected_lz") or {}).get("id")
        if selected_lz_id is not None and selected_lz_id not in {c.get("id") for c in result.get("landing_zones", [])}:
            raise HTTPException(400, "Selected candidate does not belong to this analysis")
        rid = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO reports VALUES (?, ?, ?, ?, 'PENDING', NULL, ?, ?, NULL, ?)",
                       (rid, aid, selected_lz_id or "", secrets.token_urlsafe(32), utcnow(), time.time(), selected_lz_id))
            row = db.execute("SELECT * FROM reports WHERE analysis_id=? AND selection_key=?", (aid, selected_lz_id or "")).fetchone()
            if retry and row["status"] == "FAILED":
                db.execute("UPDATE reports SET status='PENDING', error=NULL, updated_at=? WHERE report_id=? AND status='FAILED'", (time.time(), row["report_id"]))
            rid = row["report_id"]
        return self.job(rid, internal=True)

    def job(self, rid, token=None, internal=False):
        valid_id(rid)
        with self.connect() as db: row = db.execute("SELECT * FROM reports WHERE report_id=?", (rid,)).fetchone()
        if not row: raise HTTPException(404, "Report not found")
        if not internal and not secrets.compare_digest(row["access_token"], token or ""):
            raise HTTPException(403, "Report access denied")
        return dict(row)

    def metadata(self, job, include_token=False):
        data = {k: job[k] for k in ["report_id", "analysis_id", "status", "error", "created_at", "generated_at", "selected_lz_id"]}
        data.update(download_url=f"/reports/{job['report_id']}/download", preview_url=f"/reports/{job['report_id']}/preview")
        if include_token: data["access_token"] = job["access_token"]
        return data

    def artifact(self, rid, suffix):
        valid_id(rid)
        if suffix not in ("pdf", "html"): raise HTTPException(400, "Invalid report format")
        path = self.directory / f"{rid}.{suffix}"
        if path.resolve().parent != self.directory or path.is_symlink():
            raise HTTPException(400, "Invalid report path")
        return path

    def fail(self, rid, message, expected=("PENDING", "AWAITING_N8N", "RUNNING")):
        with self.connect() as db:
            db.execute("UPDATE reports SET status='FAILED', error=?, updated_at=? WHERE report_id=? AND status IN (" + ",".join("?" for _ in expected) + ")", (message, time.time(), rid, *expected))

    def generate(self, rid):
        job = self.job(rid, internal=True)
        with self.connect() as db:
            claimed = db.execute("UPDATE reports SET status='RUNNING', updated_at=? WHERE report_id=? AND status IN ('PENDING','AWAITING_N8N')", (time.time(), rid)).rowcount
        if not claimed: return self.metadata(self.job(rid, internal=True))
        try:
            record = self.analysis(job["analysis_id"], internal=True)
            job["generated_at"] = utcnow()
            html = render_html(record, job)
            pdf = self.artifact(rid, "pdf")
            html_path = self.artifact(rid, "html")
            # Random temporary names prevent concurrent attempts sharing temporary artifacts.
            temp_pdf = self.directory / (rid + "." + uuid.uuid4().hex + ".tmp")
            try:
                render_pdf(record, job, temp_pdf)
                temp_pdf.chmod(0o600)
                os.replace(temp_pdf, pdf)
            finally:
                temp_pdf.unlink(missing_ok=True)
            html_path.write_text(html, encoding="utf-8")
            html_path.chmod(0o600)
            with self.connect() as db:
                db.execute("UPDATE reports SET status='READY', generated_at=?, updated_at=?, error=NULL WHERE report_id=? AND status='RUNNING'", (job["generated_at"], time.time(), rid))
        except Exception:
            # Do not leak paths, credentials or raw exception text into client-facing errors.
            self.fail(rid, "Report generation failed. Check backend logs and retry.")
            import logging
            logging.getLogger(__name__).exception("Report rendering failed for %s", rid)
        return self.metadata(self.job(rid, internal=True))

    def dispatch(self, rid):
        url = os.getenv("N8N_REPORT_WEBHOOK_URL")
        webhook_secret = os.getenv("N8N_REPORT_WEBHOOK_SECRET")
        api_secret = os.getenv("REPORT_INTERNAL_TOKEN")
        if not webhook_secret or not api_secret:
            self.fail(rid, "n8n authentication configuration is incomplete.")
            return
        with self.connect() as db:
            claimed = db.execute("UPDATE reports SET status='AWAITING_N8N', updated_at=? WHERE report_id=? AND status='PENDING'", (time.time(), rid)).rowcount
        if not claimed: return
        job = self.job(rid, internal=True)
        try:
            with httpx.Client(timeout=15, follow_redirects=False) as client:
                response = client.post(url, headers={"X-Lunar-Webhook-Secret": webhook_secret},
                                       json={"event": "analysis.report.requested", "schema_version": 1,
                                             "analysis_id": job["analysis_id"], "report_id": rid})
                response.raise_for_status()
        except Exception:
            self.fail(rid, "n8n webhook delivery failed. Check configuration and retry.", expected=("AWAITING_N8N",))

    def tick(self):
        timeout = int(os.getenv("REPORT_JOB_TIMEOUT_SECONDS", "300"))
        with self.connect() as db:
            db.execute("UPDATE reports SET status='FAILED', error='Report job timed out or was interrupted. Retry generation.', updated_at=? WHERE status IN ('AWAITING_N8N','RUNNING') AND updated_at<?", (time.time(), time.time()-timeout))
            jobs = db.execute("SELECT report_id FROM reports WHERE status='PENDING' ORDER BY created_at LIMIT 4").fetchall()
        for job in jobs:
            self.dispatch(job[0]) if os.getenv("N8N_REPORT_WEBHOOK_URL") else self.generate(job[0])
