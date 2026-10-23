"""Upload, poll, and save a Label Studio import without exposing API tokens."""
import argparse
import json
import os
from pathlib import Path
import time
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--api", default="http://localhost:8008")
    parser.add_argument("--profile", default="human", choices=["human", "yam", "dex", "tianji"])
    parser.add_argument("--precision", choices=["nf4", "int8", "bf16_offload", "fp32_offload"], default="")
    parser.add_argument("--generation-timeout", type=int, default=0, help="Per-window generation deadline in seconds; 0 uses the mode default")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--video-url", default="")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--no-clips", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output exists: {args.output}; choose a new --output")
    headers = {}
    if token := os.environ.get("FLORA_API_TOKEN"):
        headers["Authorization"] = "Bearer " + token
    with httpx.Client(base_url=args.api.rstrip("/"), headers=headers, timeout=120) as client:
        with args.video.open("rb") as stream:
            response = client.post("/jobs", files={"video": (args.video.name, stream, "video/mp4")},
                                   data={"profile": args.profile, "instruction": args.instruction,
                                         "precision": args.precision,
                                         "generation_timeout_s": args.generation_timeout,
                                         "video_url": args.video_url, "export_clips": str(not args.no_clips).lower()})
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"Job {job_id}: {args.api}/jobs/{job_id}", flush=True)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            response = client.get(f"/jobs/{job_id}")
            response.raise_for_status()
            job = response.json()
            if job["status"] == "failed":
                raise RuntimeError(job.get("error", "Job failed"))
            if job["status"] == "completed":
                response = client.get(f"/jobs/{job_id}/label-studio")
                response.raise_for_status()
                with args.output.open("x", encoding="utf-8") as stream:
                    json.dump(response.json(), stream, indent=2)
                print(f"Saved {args.output}; {len(job['segments'])} segments")
                return
            time.sleep(2)
    raise TimeoutError(f"Job {job_id} is still running; poll it using the API")


if __name__ == "__main__":
    main()
