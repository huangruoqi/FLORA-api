"""Single GPU worker, bounded uploads, persistent job receipts and clip exports."""
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import sqlite3
import subprocess
import threading
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(os.environ.get("FLORA_DATA_DIR", ".runtime/jobs")).resolve()
MAX_BYTES = 256 * 1024 * 1024
work = queue.Queue(maxsize=8)
log = logging.getLogger("flora")
engine = None


def database():
    db = sqlite3.connect(ROOT / "jobs.sqlite3", timeout=30)
    db.row_factory = sqlite3.Row
    return db


def update(job_id, status, **values):
    with database() as db:
        db.execute("UPDATE jobs SET status=?,payload=? WHERE id=?",
                   (status, json.dumps(values), job_id))


def read(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(404, "Job not found")
    with database() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    return {"id": row["id"], "status": row["status"], **json.loads(row["payload"])}


def prediction(result):
    # TimelineLabels uses one-based inclusive FRAME numbers, not seconds.
    return {"model_version": "flora-tempseg-" + os.environ.get("FLORA_PRECISION", "nf4") + "-v1", "result": [
        {"id": f"segment_{i}", "from_name": "segments", "to_name": "video",
         "type": "timelinelabels", "value": {"ranges": [{"start": s["start_frame"], "end": s["end_frame"]}],
                                                "timelinelabels": ["Action"]}}
        for i, s in enumerate(result["segments"])]}


def make_clips(folder, result):
    import imageio_ffmpeg
    if len(result["segments"]) > 256:
        raise ValueError("Too many intervals for clip export")
    for i, segment in enumerate(result["segments"]):
        name = f"clip_{i:04d}.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
                        "-ss", str(segment["start_s"]), "-i", str(folder / "input.mp4"),
                        "-t", str(segment["end_s"] - segment["start_s"]),
                        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "fast",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(folder / name)],
                       check=True, timeout=300, capture_output=True)
        segment["clip_path"] = f"/jobs/{folder.name}/files/{name}"


def worker():
    global engine
    while True:
        job = work.get()
        try:
            if job is None:
                return
            job_id, profile, instruction, clips, video_url = job
            update(job_id, "running")
            if engine is None:
                from .engine import Engine
                engine = Engine()
            result = engine.segment(ROOT / job_id / "input.mp4", profile, instruction)
            if clips:
                make_clips(ROOT / job_id, result)
            result["prediction"] = prediction(result)
            result["label_studio_task"] = {"data": {"video": video_url or f"/jobs/{job_id}/files/input.mp4",
                                                   "flora_job_id": job_id, "fps": result["fps"]},
                                            "predictions": [result["prediction"]]}
            update(job_id, "completed", **result)
        except Exception as exc:
            log.exception("FLORA job failed")
            update(job[0], "failed", error=str(exc))
        finally:
            work.task_done()


def authorize(request: Request):
    token = os.environ.get("FLORA_API_TOKEN", "")
    if token and not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer " + token):
        raise HTTPException(401, "Bearer token required")


@asynccontextmanager
async def lifespan(app):
    ROOT.mkdir(parents=True, exist_ok=True)
    with database() as db:
        db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,status TEXT,payload TEXT)")
        db.execute("UPDATE jobs SET status='failed',payload=? WHERE status IN ('queued','running')",
                   (json.dumps({"error": "Service restarted; submit again"}),))
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    yield
    # A model call cannot be cancelled safely. Run exactly one server process.
    work.put(None)
    thread.join(timeout=1)


app = FastAPI(title="FLORA video splitting", version="0.1.0", lifespan=lifespan,
              dependencies=[Depends(authorize)])
app.add_middleware(CORSMiddleware,
                   allow_origins=os.environ.get("FLORA_CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])


@app.get("/health")
def health():
    import torch
    return {"status": "up", "cuda": torch.cuda.is_available(),
            "model_loaded": engine is not None and engine.session.model is not None,
            "queued": work.qsize(), "deployment": "experimental-" + os.environ.get("FLORA_PRECISION", "nf4")}


@app.post("/jobs", status_code=202)
async def submit(video: UploadFile = File(...), profile: str = Form("human"),
                 instruction: str = Form(""), export_clips: bool = Form(True),
                 video_url: str = Form("")):
    if profile not in {"human", "yam", "dex", "tianji"} or len(instruction) > 2000:
        raise HTTPException(422, "Invalid profile or instruction longer than 2000 characters")
    job_id = uuid4().hex
    folder = ROOT / job_id
    folder.mkdir()
    try:
        size = 0
        with (folder / "input.mp4").open("wb") as stream:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "Video exceeds 256 MiB")
                stream.write(chunk)
        if not size:
            raise HTTPException(422, "Empty video")
        with database() as db:
            db.execute("INSERT INTO jobs VALUES (?,?,?)", (job_id, "queued", "{}"))
        try:
            work.put_nowait((job_id, profile, instruction, export_clips, video_url))
        except queue.Full:
            update(job_id, "failed", error="Queue full")
            raise HTTPException(429, "GPU queue full; retry later")
    except Exception:
        shutil.rmtree(folder)
        raise
    finally:
        await video.close()
    return {"id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return read(job_id)


@app.get("/jobs/{job_id}/label-studio")
def export(job_id: str, request: Request):
    job = read(job_id)
    if job["status"] != "completed":
        raise HTTPException(409, "Job is not completed")
    task = job["label_studio_task"]
    if task["data"]["video"].startswith("/jobs/"):
        task["data"]["video"] = str(request.base_url).rstrip("/") + task["data"]["video"]
    return [task]


@app.get("/jobs/{job_id}/files/{name}")
def media(job_id: str, name: str):
    read(job_id)
    if not re.fullmatch(r"input\.mp4|clip_[0-9]{4}\.mp4", name):
        raise HTTPException(404, "File not found")
    path = ROOT / job_id / name
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4")


@app.post("/predict")
async def predict(request: Request):
    """Label Studio ML prediction endpoint for previously completed split jobs."""
    payload = await request.json()
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise HTTPException(422, "tasks must be a nonempty array")
    results = []
    for task in tasks:
        job_id = task.get("data", {}).get("flora_job_id", "")
        job = read(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, "Submit and finish /jobs before requesting predictions")
        results.append(job["prediction"])
    return {"results": results}


@app.post("/setup")
def setup():
    return {"model_version": "flora-tempseg-" + os.environ.get("FLORA_PRECISION", "nf4") + "-v1"}
