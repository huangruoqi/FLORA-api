# Setup record — this Windows PC

Date: October 22, 2026 (America/Los_Angeles). Commands run from PowerShell unless
marked Ubuntu. Repository: `https://github.com/huangruoqi/FLORA-api.git`.
No model weights, uploaded videos, credentials or virtual environments are committed.

## Initial inspection

The working folder `C:\Users\huang\OneDrive\Documents\ChatGPT\kt22-annotate`
contained an empty local Git repository. Cloned the requested repository into
its `FLORA-api` subdirectory; the remote was also empty.

```powershell
git clone https://github.com/huangruoqi/FLORA-api.git FLORA-api
cd FLORA-api
nvidia-smi
wsl --status
wsl -d Ubuntu -- bash -lc 'uname -a; python3 --version; nvidia-smi; free -h; df -h ~'
```

Observed:

- NVIDIA GeForce RTX 5080, 16,303 MiB VRAM; Windows driver 616.92.
- Windows system RAM: 33,345,101,824 bytes (about 31 GiB).
- Ubuntu is the default WSL2 distribution; GPU is exposed to Ubuntu.
- WSL kernel `6.18.33.2-microsoft-standard-WSL2`; approximately 15 GiB RAM and
  4 GiB swap available to WSL. Ubuntu initially had Python 3.14.4, no uv/FFmpeg.
- Windows `python` resolved to an unusable Microsoft Store alias.
- C: had about 1.46 TB free. WSL filesystem reported about 955 GiB free.

The driver reports CUDA 13.4 capability; this is not the installed PyTorch CUDA
runtime version. Installed CUDA 12.8 PyTorch wheels include their own runtime.
Did not install CUDA Toolkit, replace drivers, change `.wslconfig`, shut down
WSL, modify Windows firewall, install Docker, or create a public tunnel.

## Reference research and implementation

Read NVIDIA's model card, `batch_utils` API, loader, preprocessing, pipeline,
manifest checks, and model revisions from Hugging Face. NVIDIA supports Linux;
the paired workflow documents Python 3.12 and an 80 GB GPU. The released native
segmenter loads Qwen3.5-9B plus LoRA entirely onto CUDA. The Captioner is a
separate 27B model and is unnecessary for this segmentation-only stage.

Resolved the reference repository's HEAD to
`b3f079b160a23f78026b72b80fb7a0e05c57eba8`, then pinned that source commit in
`scripts/fetch_reference.py`. Reference model specifications:

| Asset | Revision |
| --- | --- |
| NVIDIA TempSeg adapter | `114a1076f70eb53f4a58ac045505db3087e79347` |
| Qwen3.5-9B base | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

Adapter SHA-256:
`1d80fe544e4cd1a7ff6b0ea4be71e988972408dab81d4c8af72e3a628739d6c3`.

Implemented a thin temporal-only wrapper around NVIDIA's native segment/stitch
pipeline. Initially tested Accelerate offloading with a 10 GiB GPU / 10 GiB CPU weight
budget, while retaining NVIDIA's GDN FP32 loader patch and BF16 compute
autocast. Added checkpoint equality checks for FP32 adapter/GDN values,
including parameters held in Accelerate's CPU/disk backing store. This changes
device placement, not the native prompts, preprocessing or stitcher source.
The first real load exposed a Transformers 5.8 / Accelerate 1.12 compatibility
issue: Transformers stores `_no_split_modules` as a set, but the PEFT redispatch
path expects a list. The wrapper converts these sets to sorted lists before
adapter loading, without changing their members. Shell files use LF line
endings enforced through `.gitattributes` so Windows clones work under WSL.

The full-precision configuration loaded and verified all 496 adapter tensors
and 48 GDN tensors, but also offloaded approximately 2.46 GB of weights to
disk. Stopped its synthetic inference test after several minutes without a
completed generation, then switched the default to **NF4 4-bit base-model
linear layers**, leaving the vision encoder/output head unquantized, with BF16
compute and FP32 adapter/GDN values. This changes base-model precision and may
change segmentation accuracy. The BF16 offload option remains available but
was not validated to completion. Installed the official CUDA-capable package:

```bash
.tools/uv pip install --python .runtime/venv/bin/python bitsandbytes==0.50.2
```

During loading, a concurrent pytest process reported `Cannot allocate memory`.
Stopped concurrent model loading and reran tests serially: all six passed.
No WSL or Windows memory configuration was changed.

Added a FastAPI service with bounded upload/queue handling, persistent SQLite
job status, a single GPU worker, FFmpeg H.264 video clip generation, authenticated
file endpoints, Label Studio imports, and prediction retrieval for completed jobs.
Used HumanSignal's official tag docs and converter implementation to confirm
TimelineLabels uses one-based inclusive frame ranges. Only CFR uploads are
accepted for this integration so Label Studio's frame timeline stays consistent.

## Install performed

Bootstrapped uv 0.13.5 under `.tools` inside the repository, using the installer
from `https://astral.sh/uv/install.sh`. The repeatable installer is now:

```powershell
wsl -d Ubuntu -- bash scripts/setup.sh
```

The script installs Python 3.12 under `.runtime/python`, creates
`.runtime/venv`, installs PyTorch 2.11.0+cu128 / torchvision 0.26.0 using
`https://download.pytorch.org/whl/cu128`, then installs `requirements.txt` and
downloads pinned reference code. Observed installed Python: **3.12.15**.

Installation output: `.runtime-setup.log` (local, ignored by Git).
Installing and first importing libraries from the Windows/OneDrive mount took
several minutes. Setup was rerun after an edit to the running shell script
interrupted its final steps; the rerun reused installed packages and completed.

GPU verification, Ubuntu:

```bash
.runtime/venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Observed: `2.11.0+cu128 True`, `NVIDIA GeForce RTX 5080`.

## Verification performed

Ubuntu:

```bash
PYTHONPATH=.runtime/reference:. .runtime/venv/bin/python -m pytest -q
```

Result: **6 passed**. Tests cover queued completion and failed jobs, input
validation, token enforcement, Label Studio frame-range output and prediction
retrieval, file restrictions, and real H.264 extraction from a generated video.
Tests mock model inference; they do not establish FLORA model quality.
One upstream Starlette/httpx deprecation warning was emitted.

Started the actual server from the repository directory:

```powershell
wsl -d Ubuntu -- bash scripts/start.sh
Invoke-RestMethod http://localhost:8008/health
```

Windows localhost access was verified. The response reported `status: up`,
`cuda: true`, `model_loaded: false`, `queued: 0` before any API model job.
The OpenAPI interface is `http://localhost:8008/docs`.

Real model smoke test, Ubuntu:

```bash
HF_HOME=.runtime/huggingface FLORA_MODEL_CACHE=.runtime/models \
  PYTHONPATH=.runtime/reference:. .runtime/venv/bin/python -u scripts/smoke_model.py \
  > .runtime/model-smoke.log 2>&1
```

The test generated a four-second 24 FPS video and downloaded pinned model assets.
The BF16 offload run was stopped as described above. A generated test pattern
can verify execution, not labeling accuracy.

The user then supplied a real sample. Validated it with
`DataEnginePipeline().inspect_inputs(...)` before inference:

- File: `C:\Users\huang\Downloads\0ac0a71f-98e3-4b09-3273-6d19ede20a7d.mp4`.
- 5,663,567 bytes; 24.0 seconds; 30 FPS CFR; 720 frames; 1280 × 970 pixels.
- SHA-256: `69affe0faaa28f58a5defed9df3ebff6072003b47f2b2cceaa0e332b556c07dc`.

Submitted through the actual running API using the client, Ubuntu:

```bash
.runtime/venv/bin/python -u scripts/split_video.py \
  /mnt/c/Users/huang/Downloads/0ac0a71f-98e3-4b09-3273-6d19ede20a7d.mp4 \
  --output .runtime/sample-label-studio.json
```

Job ID: `a951aa5ccc9a427d836ad44dca036be0`.
Local client log: `.runtime/sample-client.log`.
Sample media and predictions remain local, ignored by Git.

**Result: completed successfully with `experimental-nf4`.** Four native inference
windows (including adaptive processing) produced these five source intervals:

| Clip | Start seconds | End seconds | Label Studio frames |
| --- | --- | --- | --- |
| `clip_0000.mp4` | 0.00 | 4.60 | 1–138 |
| `clip_0001.mp4` | 4.60 | 13.65 | 139–410 |
| `clip_0002.mp4` | 4.60 | 15.25 | 139–458 |
| `clip_0003.mp4` | 13.65 | 15.25 | 411–458 |
| `clip_0004.mp4` | 15.25 | 24.00 | 459–720 |

Intervals overlap because the native model/stitcher retains concurrent events.
These are proposed temporal annotations and have not been assessed against
human ground truth. No Captioner calls were made.

Verified by decoding all five files with PyAV: H.264, nonempty frames, and
durations within two source frames of the requested intervals. Verified the
saved Label Studio JSON contains five frame-based predictions with `fps=30`.
Actual `/predict` retrieval returned five results. A Windows HTTP byte-range
request for the first clip returned **206 Partial Content**. Final `/health`
returned `cuda: true`, `model_loaded: true`, `queued: 0`.
The Label Studio browser UI itself was not available for an interactive import
test; use the supplied XML and JSON in your project.

Cold model initialization from the OneDrive mount took several minutes. The
first model generation took 22.411 seconds; the next took 5.279 seconds. Peak
CUDA tensor allocation reported by the model was 11.138 GiB, excluding Windows
display usage and other non-PyTorch allocations. These timings are observations,
not a throughput or annotation-quality benchmark.

The API is left running locally on port 8008 with the NF4 model loaded. It is
not registered for automatic startup. Restart after stopping/rebooting using
`wsl -d Ubuntu -- bash scripts/start.sh` from this repository directory.

## Split review webpage follow-up

Added `/` and `/preview` to the existing local API, plus an authenticated
`GET /jobs` summary for selecting recent results. The page displays the original
video, one timeline lane per proposed interval, playable clip cards, downloads,
and Label Studio JSON export. Overlapping intervals remain visible on separate
lanes. Static page access does not expose job data; API and media authorization
still apply. Tokens entered in the page remain in browser memory only.

Restarted the local API to load the page. Existing jobs/clips remain on disk;
the model loads lazily on the next inference request after a server restart.
Verified the webpage in the in-app browser: the sample's five intervals and
clip thumbnails render, selecting split 02 updates the selection to
4.60–13.65 seconds / frames 139–410, and the selected clip plays. API tests also
cover public shell access with protected job-list access when a token is set.

## Sources

- [NVIDIA model card](https://huggingface.co/nvidia/Flora-TempSeg)
- [NVIDIA inference instructions](https://huggingface.co/nvidia/Flora-TempSeg/blob/b3f079b160a23f78026b72b80fb7a0e05c57eba8/batch_utils/README.md)
- [NVIDIA noncommercial model license](https://huggingface.co/nvidia/Flora-TempSeg/blob/main/LICENSE.pdf)
- [Label Studio TimelineLabels](https://labelstud.io/tags/timelinelabels)
- [HumanSignal frame conversion implementation](https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/examples/yolo/utils/converter.py)
