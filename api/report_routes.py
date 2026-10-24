import asyncio
import secrets
import os
from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from api.report_service import ReportService


def create_router(service: ReportService):
    router = APIRouter()

    def internal_auth(x_lunar_report_token: Optional[str] = Header(None)):
        expected = os.getenv("REPORT_INTERNAL_TOKEN")
        if not expected: raise HTTPException(503, "Report orchestration is not configured")
        if not secrets.compare_digest(expected, x_lunar_report_token or ""):
            raise HTTPException(401, "Invalid orchestration credential")

    class GenerateRequest(BaseModel):
        analysis_id: str
        analysis_token: str
        selected_lz_id: Optional[str] = None
        retry: bool = False

    class RenderRequest(BaseModel):
        analysis_id: str
        report_id: str

    @router.post("/reports/generate", status_code=202)
    def generate(body: GenerateRequest):
        job = service.create(body.analysis_id, body.analysis_token, body.selected_lz_id, body.retry)
        return service.metadata(job, include_token=True)

    @router.get("/reports/{report_id}")
    def status(report_id: str, access_token: str = Query(...)):
        return service.metadata(service.job(report_id, access_token))

    def ready_file(report_id, access_token, suffix):
        job = service.job(report_id, access_token)
        if job["status"] != "READY": raise HTTPException(409, "Report is not ready")
        path = service.artifact(report_id, suffix)
        if not path.is_file(): raise HTTPException(404, "Report artifact is unavailable")
        response = FileResponse(path, media_type="application/pdf" if suffix == "pdf" else "text/html",
                               filename=f"lunar-analysis-{report_id}.pdf" if suffix == "pdf" else None)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if suffix == "html":
            response.headers["Content-Security-Policy"] = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'self'"
        return response

    @router.get("/reports/{report_id}/download")
    def download(report_id: str, access_token: str = Query(...)):
        return ready_file(report_id, access_token, "pdf")

    @router.get("/reports/{report_id}/preview")
    def preview(report_id: str, access_token: str = Query(...)):
        return ready_file(report_id, access_token, "html")

    @router.get("/reports/internal/analyses/{analysis_id}", dependencies=[Depends(internal_auth)])
    def analysis_record(analysis_id: str):
        record = service.analysis(analysis_id, internal=True)
        # No binary payloads, filesystem paths or capability tokens travel through n8n.
        r = record["result"]
        return {"analysis_id": analysis_id, "status": record["status"], "timestamp": r.get("timestamp"),
                "stages": record["stages"], "candidate_count": len(r.get("landing_zones") or []),
                "result_reference": {"storage": "backend", "analysis_id": analysis_id}}

    @router.post("/reports/internal/render", dependencies=[Depends(internal_auth)])
    def render(body: RenderRequest):
        job = service.job(body.report_id, internal=True)
        if job["analysis_id"] != body.analysis_id: raise HTTPException(409, "Report and analysis IDs do not match")
        return service.generate(body.report_id)

    @router.get("/reports/internal/jobs/{report_id}", dependencies=[Depends(internal_auth)])
    def internal_status(report_id: str):
        return service.metadata(service.job(report_id, internal=True))

    return router


def install_reports(app):
    service = ReportService()
    app.include_router(create_router(service))
    app.state.report_service = service

    async def worker():
        import logging
        while True:
            try:
                await asyncio.to_thread(service.tick)
            except Exception:
                logging.getLogger(__name__).exception("Report worker iteration failed")
            await asyncio.sleep(2)

    @app.on_event("startup")
    async def start_reports():
        app.state.report_worker = asyncio.create_task(worker())

    @app.on_event("shutdown")
    async def stop_reports():
        app.state.report_worker.cancel()
        try: await app.state.report_worker
        except asyncio.CancelledError: pass

    return service
