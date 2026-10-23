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
uses GPU/CPU/disk offloading. A separate `fp32_offload` option loads all base
weights as FP32 and disables BF16 compute autocast. Both unquantized modes can
be considerably slower on a 16 GB GPU.
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
Open [the split review webpage](http://localhost:8008/preview) to review an
existing job with the original video, clickable timeline, clip previews,
downloads, and Label Studio export. Select a recent job or use
`/preview?job=JOB_ID`. If authentication is enabled, enter the API token in the
page; it stays in memory and is not saved in browser storage.
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

Generated runtime files and videos are ignored by Git. The start script uses
`~/.cache/flora-api/` on the Linux filesystem for model downloads and offload
files, avoiding slow repeated reads through the Windows/OneDrive mount.
The Python environment, reference source, videos and SQLite receipts remain
project-local. Set the cache variables below to override these locations.
Setup does not alter your WSL configuration.

## Submit and poll

For another computer on the same router (Ethernet or Wi-Fi), use the standalone
[Python LAN client](examples/lan_client.py) and [server/client setup guide](examples/README.md).
It can upload, poll, resume, and download clips plus Label Studio JSON. The guide
includes optional Windows LAN forwarding; the localhost API is not remotely
accessible until that setup is applied.

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
`windows` counts model inference calls on overlapping temporal windows,
including adaptive retry windows. It is not the number of final splits/clips.

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

## Precision per request

Add the multipart `precision` field to `POST /jobs`. Omitting it uses
`FLORA_PRECISION` (default `nf4`).

| Value | Weight / compute precision | Placement |
| --- | --- | --- |
| `nf4` | 4-bit base linear layers, BF16 compute; FP32 GDN/adapter | GPU |
| `int8` | LLM.int8 base linear layers and output head; BF16 vision encoder; FP32 GDN/adapter | GPU |
| `bf16_offload` | Unquantized BF16 base; FP32 GDN/adapter; BF16 compute | GPU / CPU / disk |
| `fp32_offload` | Unquantized FP32 base/GDN/adapter; FP32 compute | GPU / CPU / disk |

`bf16_offload` matches NVIDIA's reference precision; “unquantized” does not mean
all parameters are FP32. Choose `fp32_offload` for actual 32-bit weights and
compute. The all-FP32 mode is experimental and has not been GPU-benchmarked on
this PC. More precision is not a guarantee of better temporal boundaries.

```powershell
curl.exe -X POST http://localhost:8008/jobs -F "video=@C:/videos/demo.mp4" -F "precision=bf16_offload"
# Reuse an already uploaded video; original job/results remain intact:
curl.exe -X POST http://localhost:8008/jobs/JOB_ID/rerun -H "Content-Type: application/json" -d '{"precision":"bf16_offload"}'
```

In the preview webpage, choose **Run precision** and click **Run this video
again**. Each rerun creates a separate job and automatically polls its status.
The worker unloads the previous model before loading a different precision;
two models are never intentionally held on the GPU together. Consecutive jobs
with the same precision reuse the loaded model. Precision selection does not
change process-wide environment settings.

Choose **8-bit · INT8** in the page, send `precision=int8` to `POST /jobs`, or
use CLI `--precision int8`. INT8 uses bitsandbytes LLM.int8 with its default
outlier threshold of 6.0. Its output head is quantized to leave activation
headroom on the 16 GB GPU; NF4 keeps that head unquantized. The vision encoder
remains BF16 and adapter/GDN checkpoint values are verified as FP32. INT8 is
configured to remain on the GPU and does not silently fall back to NF4 or CPU
offloading if it runs out of memory. Review actual test results in the setup log.

Receipts, results and Label Studio model versions record the selected precision.
Completed jobs expose `elapsed_s` (processing including model load and clip
export), `model_load_s`, and `segmentation_s`. `GET /health` reports default,
supported and currently loaded precisions separately.
Results also include `cuda_peak_allocated_gib`, reset per job, for PyTorch's peak
GPU tensor allocation; driver/display usage and the CUDA allocator reserve are
additional memory consumers.

Slow offloaded generation needs more than NVIDIA's original 300-second window
deadline on this PC. Mode defaults are 300 seconds/window for NF4 and INT8, 900 for
unquantized BF16, and 1800 for FP32. Override using the multipart or rerun JSON
field `generation_timeout_s` (30–3600 seconds). The CLI flag is
`--generation-timeout`; increase its separate overall polling `--timeout` for
long offloaded runs. The deadline is checked between generation steps and is
not a hard wall-clock cancellation guarantee. Partial model outputs are rejected.

## Label Studio

An Ubuntu client can perform upload, polling and import-file export in one step:

```bash
.runtime/venv/bin/python scripts/split_video.py /mnt/c/videos/demo.mp4 --output predictions.json
```

Use `--video-url` for an existing accessible media URL, `--no-clips` to skip clip
encoding, or `--timeout` to increase the default 30-minute polling deadline.
Use `--precision bf16_offload` or `--precision fp32_offload` for unquantized runs.
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
| `FLORA_PRECISION` | `nf4` | Default request precision: `nf4`, `int8`, `bf16_offload`, `fp32_offload` |
| `FLORA_GPU_MEMORY` | `10GiB` | Accelerate GPU weight-placement budget |
| `FLORA_CPU_MEMORY` | `12GiB` | CPU weight-placement budget |
| `HF_HOME` | `~/.cache/flora-api/huggingface` | Hugging Face cache (start script) |
| `FLORA_MODEL_CACHE` | `~/.cache/flora-api/models` | NVIDIA model snapshots (start script) |
| `FLORA_OFFLOAD_DIR` | `~/.cache/flora-api/offload` | Accelerate offload root, with one subfolder per precision |
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
HF_HOME="$HOME/.cache/flora-api/huggingface" FLORA_MODEL_CACHE="$HOME/.cache/flora-api/models" \
  PYTHONPATH="$PWD/.runtime/reference:$PWD" .runtime/venv/bin/python scripts/smoke_model.py
```

For out-of-memory failures, stop the server and close other GPU applications.
NF4 and INT8 place all weights on the GPU; the memory-budget variables
apply to the two offload modes. In those modes, reducing `FLORA_GPU_MEMORY` reserves
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
