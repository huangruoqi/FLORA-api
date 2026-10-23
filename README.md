# FLORA video splitting API for Windows + RTX 5080

Runs NVIDIA **Flora-TempSeg** in Ubuntu on WSL2. Upload a video, poll a job,
download H.264 clips, and import temporal predictions into Label Studio.
This is the segmentation step only: the 27B Captioner is not downloaded or run.

**Hardware status and exact commands used on this PC:** [setup log](docs/SETUP-LOG.md).

The RTX 5080 has 16 GB VRAM. NVIDIA's native segmenter loader places the entire
9B model on CUDA, which does not fit here at native precision. This service
defaults to bitsandbytes NF4 4-bit base-model linear layers with BF16 compute,
while retaining FP32 Gated DeltaNet/LoRA values and leaving the vision encoder
and output head unquantized. This is an experimental deployment adaptation,
not NVIDIA's validated configuration; quantization can change predictions.
An optional `FLORA_PRECISION=bf16_offload` mode retains the unquantized base and
uses GPU/CPU/disk offloading, but was too slow in the initial test on this PC.
`/health` distinguishes an available API from a loaded model. Check the setup
log for what has actually been verified.

## Install

Requirements: Windows NVIDIA driver exposing the GPU to WSL2, Ubuntu, outbound
HTTPS, approximately 40 GB free disk for packages/model/cache and additional
space for videos. The current PC has 32 GB RAM; WSL currently sees about 15 GB.
No Windows CUDA toolkit or Linux NVIDIA driver installation is necessary.

From PowerShell in this repository:

```powershell
wsl -d Ubuntu -- nvidia-smi
wsl -d Ubuntu -- bash scripts/setup.sh
```

Run commands below in Ubuntu, from the repository directory:

```bash
bash scripts/start.sh
```

Windows can access [interactive API docs](http://localhost:8008/docs) and
[health](http://localhost:8008/health) through WSL localhost forwarding.
The server uses one process and one worker thread to serialize GPU calls.
Do not start multiple Uvicorn workers against this data directory.

The installer creates a project-local Python 3.12 environment and installs CUDA
12.8 PyTorch wheels, pinned model dependencies, PyAV and bundled FFmpeg.
It fetches NVIDIA's reference code at commit
`b3f079b160a23f78026b72b80fb7a0e05c57eba8` into `.runtime/reference`.
Model weights are fetched on the first real job, with base/adapter revisions and
checksums selected by that reference's `batch_utils/models.json`.
NVIDIA's vendor-code manifest and model manifests are verified before inference.
The adapter's FP32 values and GDN values are checked at load time.

All generated runtime files and videos are ignored by Git. Do not put the
model cache in a cloud-sync folder if avoidable: set `HF_HOME`,
`FLORA_MODEL_CACHE`, `FLORA_OFFLOAD_DIR`, and `FLORA_DATA_DIR` to Linux disk
locations for better I/O. Setup currently defaults to project-local paths for
reproducibility and does not alter your WSL configuration.

## Submit and poll

PowerShell (use `curl.exe`, not the older PowerShell curl alias):

```powershell
curl.exe -X POST http://localhost:8008/jobs -F "video=@C:/videos/demo.mp4" -F "profile=human" -F "export_clips=true"
curl.exe http://localhost:8008/jobs/JOB_ID
curl.exe http://localhost:8008/jobs/JOB_ID/label-studio -o predictions.json
```

Upload response is HTTP 202 with `id` and `status_url`. Poll until `completed`
or `failed`. Completed results include source-relative `start_s`/`end_s`,
one-based frame ranges, provenance from the native stitcher, clip download
paths, model revisions and source SHA-256. `clip_path` is relative to the API
origin. FFmpeg re-encodes video-only H.264 clips, without audio. Operations can
overlap; clips can overlap and gaps can remain. No synthetic cuts fill gaps.

Set `export_clips=false` to return just boundaries/predictions. Optional
`instruction` supplies an episode task (maximum 2000 characters). Optional
`video_url` supplies a video URL already accessible to Label Studio and its
browser; it is copied into the import without being fetched by this service.

Input limits: one camera, constant frame rate, at least 4 FPS, at most 120
seconds, 256 MiB, 4096 pixels per side and 7200 frames, per NVIDIA's media
validator. Profiles: `human`, `yam`, `dex`, `tianji`. Three-camera and long-window
requests are not exposed by this first version. For VFR inputs, normalize to
CFR first, then upload that exact normalized file and label that same file.
Timing predictions remain approximate and need human review.

## Label Studio

An Ubuntu client can perform upload, polling and import-file export in one step:

```bash
.runtime/venv/bin/python scripts/split_video.py /mnt/c/videos/demo.mp4 --output predictions.json
```

Use `--video-url` for an existing accessible media URL, `--no-clips` to skip clip
encoding, or `--timeout` to increase the default 30-minute polling deadline.
The client reads `FLORA_API_TOKEN` from the environment and preserves existing
output files.

1. Copy [label-studio.xml](label-studio.xml) into the project's labeling interface.
2. Submit the video to `/jobs`; wait for completion.
3. Download `/jobs/JOB_ID/label-studio` and import the JSON into Label Studio.
4. Review the `Action` spans. The separate captioning stage can attach action
   descriptions later; this service does not invent names for the operations.

The import uses `data.video`, `data.fps`, `data.flora_job_id` and `predictions`.
`TimelineLabels` ranges are **one-based inclusive frame numbers**, not seconds;
the adapter maps model intervals to actual source presentation timestamps.
See the [official tag documentation](https://labelstud.io/tags/timelinelabels)
and [HumanSignal's frame-range converter](https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/examples/yolo/utils/converter.py).

`POST /setup`, `GET /health` and `POST /predict` also support a narrow ML-backend
workflow: `/predict` takes `{"tasks":[{"data":{"flora_job_id":"JOB_ID"}}]}`
and returns completed predictions. It does not automatically retrieve task URLs
or schedule inference. Your workflow must upload and complete the split job
first; unfinished jobs return HTTP 409.

If Label Studio runs in Docker, its backend cannot reach the host using its own
`localhost`: use `host.docker.internal` where supported. Imported video URLs
must also resolve from the annotator's browser. Provide `video_url` for your
existing media storage when these addresses differ. For remote Label Studio,
configure an authenticated HTTPS reverse proxy and accessible media storage.

## Configuration and operation

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLORA_HOST` | `127.0.0.1` | Bind address |
| `FLORA_PORT` | `8008` | API port |
| `FLORA_API_TOKEN` | unset | Optional Bearer authentication on all routes |
| `FLORA_CORS_ORIGINS` | localhost / 127.0.0.1 on port 8080 | Comma-separated Label Studio browser origins |
| `FLORA_PRECISION` | `nf4` | `nf4` (GPU-resident 4-bit) or `bf16_offload` (experimental, slow) |
| `FLORA_GPU_MEMORY` | `10GiB` | Accelerate GPU weight-placement budget |
| `FLORA_CPU_MEMORY` | `10GiB` | CPU weight-placement budget |
| `HF_HOME` | `.runtime/huggingface` | Hugging Face cache |
| `FLORA_MODEL_CACHE` | `.runtime/models` | NVIDIA model snapshots |
| `FLORA_OFFLOAD_DIR` | `.runtime/offload` | Accelerate disk offload |
| `FLORA_DATA_DIR` | `.runtime/jobs` | SQLite receipts, original videos, clips |

Set variables in Ubuntu before `bash scripts/start.sh`. When setting an API
token, send `Authorization: Bearer ...` on requests. Browser video elements
cannot automatically attach that header; use `video_url` pointing to your
existing media store, or an authenticated proxy that handles media access.
Do not put tokens in source control or video URLs. Loopback is the default;
the setup does not open a Windows firewall port or public tunnel.

Queue capacity is eight waiting jobs. Full queues return HTTP 429. Job receipts
survive restarts; interrupted queued/running jobs are marked failed and must be
resubmitted. Videos/clips remain on disk until manually removed while the server
is stopped; there is no retention scheduler. Stop with Ctrl+C. Restart with
`bash scripts/start.sh`. This version does not install an automatic startup service.

## Verify

```bash
PYTHONPATH="$PWD/.runtime/reference:$PWD" .runtime/venv/bin/python -m pytest -q
# Real model download + GPU execution; synthetic video is not an accuracy benchmark:
HF_HOME="$PWD/.runtime/huggingface" FLORA_MODEL_CACHE="$PWD/.runtime/models" \
  PYTHONPATH="$PWD/.runtime/reference:$PWD" .runtime/venv/bin/python scripts/smoke_model.py
```

For out-of-memory failures, stop the server and close other GPU applications.
The default NF4 mode places all weights on the GPU; the memory-budget variables
apply only to `bf16_offload`. In that mode, reducing `FLORA_GPU_MEMORY` reserves
activation memory by shifting more weights to RAM/disk, which slows inference.
If Hugging Face returns 401/403, obtain model access and set `HF_TOKEN` in the
server environment, then restart. The API never returns this token.

## Model license and sources

NVIDIA designates Flora for **non-commercial robotics research and development**.
Its model is governed by the
[NVIDIA OneWay Noncommercial License](https://huggingface.co/nvidia/Flora-TempSeg/blob/main/LICENSE.pdf).
The base Qwen model has its own Apache-2.0 license. Review NVIDIA's
[model card](https://huggingface.co/nvidia/Flora-TempSeg),
[reference pipeline](https://huggingface.co/nvidia/Flora-TempSeg/blob/main/batch_utils/README.md)
and downloaded third-party notices before use. Upstream code, model weights and
licenses are fetched into `.runtime/reference` rather than relicensed here.
Quantization uses the official [bitsandbytes CUDA package](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation).
