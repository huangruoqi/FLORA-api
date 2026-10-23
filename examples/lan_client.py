"""FLORA client for a second computer on the same LAN (Ethernet or Wi-Fi).

Install: python -m pip install -r examples/requirements.txt
Submit:  python examples/lan_client.py --video demo.mp4 --precision int8
Resume:  python examples/lan_client.py --job-id JOB_ID
Token:   set FLORA_API_TOKEN in the environment when the server requires it.
See examples/README.md for the Windows/WSL LAN forwarding step.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time

import httpx


def checked(response):
    if response.status_code == 401:
        raise RuntimeError("Authentication rejected. Set FLORA_API_TOKEN to match the server.")
    response.raise_for_status()
    return response


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def download_clip(client, job_id, index, segment, folder):
    expected = f"/jobs/{job_id}/files/clip_{index:04d}.mp4"
    if not segment.get("clip_path"):
        return  # The job was submitted with export_clips=false.
    if segment["clip_path"] != expected:
        raise RuntimeError("Unexpected clip URL in server response")
    path = folder / f"clip_{index:04d}.mp4"
    if path.exists():
        print(f"Keeping existing {path.name}")
        return
    temporary = path.with_suffix(".mp4.part")
    with client.stream("GET", expected.lstrip("/")) as response:
        checked(response)
        with temporary.open("wb") as stream:
            for chunk in response.iter_bytes():
                stream.write(chunk)
    temporary.replace(path)
    print(f"Downloaded {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://10.0.0.141:8009",
                        help="Ethernet-connected server's LAN bridge URL")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Video path on THIS client computer")
    source.add_argument("--job-id", help="Poll/download an already submitted job without uploading again")
    source.add_argument("--check", action="store_true", help="Only check server connectivity and health")
    parser.add_argument("--precision", choices=["nf4", "int8", "bf16_offload", "fp32_offload"], default="nf4")
    parser.add_argument("--profile", choices=["human", "yam", "dex", "tianji"], default="human")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--video-url", default="", help="Optional media URL accessible to Label Studio")
    parser.add_argument("--generation-timeout", type=int, default=0, help="Seconds per window; 0 uses the mode default")
    parser.add_argument("--wait-timeout", type=float, default=7200, help="Total polling time in seconds")
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("flora-results"))
    parser.add_argument("--no-download-clips", action="store_true")
    args = parser.parse_args()
    if args.poll_interval <= 0 or args.wait_timeout <= 0:
        parser.error("Polling interval and wait timeout must be positive")
    if args.video and not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    if args.job_id and not re.fullmatch(r"[a-f0-9]{32}", args.job_id):
        parser.error("job-id must be a 32-character hexadecimal identifier")
    token = os.environ.get("FLORA_API_TOKEN", "")
    headers = {"Authorization": "Bearer " + token} if token else {}
    # Disable environment HTTP proxies for this explicitly selected local server.
    timeout = httpx.Timeout(300, connect=10)
    try:
        with httpx.Client(base_url=args.server.rstrip("/") + "/", headers=headers,
                          timeout=timeout, trust_env=False) as client:
            health = checked(client.get("health")).json()
            print(f"Connected to {args.server}; CUDA={health['cuda']}")
            if args.check:
                print(json.dumps(health, indent=2))
                return
            if args.video:
                with args.video.open("rb") as stream:
                    receipt = checked(client.post("jobs", files={
                        "video": (args.video.name, stream, "video/mp4")}, data={
                        "precision": args.precision, "profile": args.profile,
                        "instruction": args.instruction, "export_clips": "true",
                        "video_url": args.video_url,
                        "generation_timeout_s": args.generation_timeout})).json()
                job_id = receipt["id"]
            else:
                job_id = args.job_id
                receipt = {"id": job_id}
            if not re.fullmatch(r"[a-f0-9]{32}", job_id):
                raise RuntimeError("Invalid job identifier returned by server")
            folder = args.output_dir / job_id
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "receipt.json").exists():
                save_json(folder / "receipt.json", receipt)
            print(f"Job: {job_id}\nPreview: {args.server.rstrip('/')}/preview?job={job_id}", flush=True)
            deadline = time.monotonic() + args.wait_timeout
            last_status = None
            while True:
                job = checked(client.get(f"jobs/{job_id}")).json()
                if job["status"] != last_status:
                    print(f"Status: {job['status']} · {job.get('precision', 'unknown')}", flush=True)
                    last_status = job["status"]
                if job["status"] in {"completed", "failed"}:
                    save_json(folder / "result.json", job)
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Still processing. Resume with --job-id {job_id}; the server job continues.")
                time.sleep(args.poll_interval)
            if job["status"] == "failed":
                raise RuntimeError(f"Job failed: {job.get('error', 'unknown error')}")
            task = checked(client.get(f"jobs/{job_id}/label-studio")).json()
            save_json(folder / "label-studio.json", task)
            if not args.no_download_clips:
                for i, segment in enumerate(job["segments"]):
                    download_clip(client, job_id, i, segment, folder)
            print(f"Done: {len(job['segments'])} splits; saved in {folder.resolve()}")
            print(f"Server processing: {job.get('elapsed_s', 'not recorded')} seconds")
    except httpx.ConnectError as exc:
        raise SystemExit(f"Cannot reach {args.server}. Check the LAN forwarding/firewall steps in examples/README.md.") from exc
    except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
