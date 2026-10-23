import importlib
import time
from types import SimpleNamespace
from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("FLORA_DATA_DIR", str(tmp_path / "jobs"))
    monkeypatch.delenv("FLORA_API_TOKEN", raising=False)
    monkeypatch.delenv("FLORA_PRECISION", raising=False)
    import flora_api.app as module
    m = importlib.reload(module)
    class FakeEngine:
        precision = "nf4"
        session = SimpleNamespace(model=None)
        def segment(self, path, profile, instruction):
            if path.read_bytes() == b"bad":
                raise ValueError("invalid video")
            return {"duration_s": 2, "fps": 24, "segments": [
                {"start_s": .5, "end_s": 1.5, "start_frame": 13, "end_frame": 36}]}
    monkeypatch.setattr(m, "engine", FakeEngine())
    with TestClient(m.app) as client:
        yield client, m


def finish(client, job_id):
    for _ in range(200):
        response = client.get(f"/jobs/{job_id}").json()
        if response["status"] in {"completed", "failed"}:
            return response
        time.sleep(.01)
    raise AssertionError("Worker did not complete")


def test_job_and_label_studio(api):
    client, m = api
    response = client.post("/jobs", files={"video": ("video.mp4", b"video")},
                           data={"export_clips": "false"})
    assert response.status_code == 202
    job_id = response.json()["id"]
    job = finish(client, job_id)
    assert job["status"] == "completed"
    summary = client.get("/jobs").json()
    assert summary == [{"id": job_id, "status": "completed", "duration_s": 2, "precision": "nf4", "segments": 1}]
    result = job["prediction"]["result"][0]
    assert result["value"]["ranges"] == [{"start": 13, "end": 36}]
    task = client.get(f"/jobs/{job_id}/label-studio").json()[0]
    assert task["data"]["video"].startswith("http://testserver/jobs/")
    assert task["data"]["fps"] == 24
    assert client.post("/predict", json={"tasks": [task]}).json()["results"] == [job["prediction"]]
    assert client.get(f"/jobs/{job_id}/files/input.mp4").content == b"video"
    assert client.get(f"/jobs/{job_id}/files/jobs.sqlite3").status_code == 404


def test_failed_job_and_input_validation(api):
    client, _ = api
    assert client.post("/jobs", files={"video": ("v.mp4", b"")}).status_code == 422
    assert client.post("/jobs", files={"video": ("v.mp4", b"x")}, data={"profile": "bad"}).status_code == 422
    job = client.post("/jobs", files={"video": ("v.mp4", b"bad")}, data={"export_clips": "false"}).json()
    assert finish(client, job["id"])["status"] == "failed"
    assert client.get(f"/jobs/{job['id']}/label-studio").status_code == 409
    assert client.get("/jobs/invalid").status_code == 404
    assert client.post("/jobs", files={"video": ("v.mp4", b"x")}, data={"precision": "invalid"}).status_code == 422
    assert client.post("/jobs", files={"video": ("v.mp4", b"x")}, data={"generation_timeout_s": "1"}).status_code == 422


def test_precision_switch_and_rerun_preserve_original(api, monkeypatch):
    client, m = api
    import flora_api.engine as engine_module
    closed = []
    class PrecisionEngine:
        session = SimpleNamespace(model=None)
        def __init__(self, precision):
            self.precision = precision
        def close(self):
            closed.append(self.precision)
        def segment(self, path, profile, instruction):
            assert path.read_bytes() == b"video"
            return {"duration_s": 2, "fps": 24, "segments": []}
    monkeypatch.setattr(engine_module, "Engine", PrecisionEngine)
    monkeypatch.setattr(m, "engine", None)
    first = client.post("/jobs", files={"video": ("v.mp4", b"video")},
                        data={"precision": "nf4", "export_clips": "false"}).json()
    assert finish(client, first["id"])["precision"] == "nf4"
    second = client.post(f"/jobs/{first['id']}/rerun", json={"precision": "bf16_offload"}).json()
    result = finish(client, second["id"])
    assert result["status"] == "completed"
    assert result["precision"] == "bf16_offload"
    assert result["request"]["generation_timeout_s"] == 900
    assert result["prediction"]["model_version"] == "flora-tempseg-bf16_offload-v1"
    assert closed == ["nf4"]
    assert client.get(f"/jobs/{first['id']}").json()["precision"] == "nf4"
    third = client.post(f"/jobs/{second['id']}/rerun", json={"precision": "fp32_offload"}).json()
    assert finish(client, third["id"])["precision"] == "fp32_offload"
    assert closed == ["nf4", "bf16_offload"]
    assert m.engine.session.generation_timeout_s == 1800
    assert client.post(f"/jobs/{first['id']}/rerun", json={"precision": "bad"}).status_code == 422


def test_token(api, monkeypatch):
    client, _ = api
    monkeypatch.setenv("FLORA_API_TOKEN", "test-secret")
    assert client.get("/preview").status_code == 200
    assert client.get("/jobs").status_code == 401
    assert client.get("/jobs", headers={"Authorization": "Bearer test-secret"}).json() == []
    assert client.post("/setup").status_code == 401
    assert client.post("/setup", headers={"Authorization": "Bearer test-secret"}).status_code == 200


def test_real_clip_export(api, tmp_path):
    import imageio_ffmpeg
    import subprocess
    import av
    _, m = api
    folder = tmp_path / ("a" * 32)
    folder.mkdir()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=duration=2:size=128x96:rate=24", "-c:v", "libx264",
                    str(folder / "input.mp4")], check=True)
    result = {"segments": [{"start_s": .5, "end_s": 1.5}]}
    m.make_clips(folder, result)
    with av.open(str(folder / "clip_0000.mp4")) as media:
        assert len(list(media.decode(video=0))) == 24
