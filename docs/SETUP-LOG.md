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

## Per-job precision follow-up

Added a multipart `precision` field to `POST /jobs` and a JSON precision option
to `POST /jobs/{job_id}/rerun`. Supported values are `nf4`, `bf16_offload`
(NVIDIA's native unquantized BF16 base / FP32 GDN+LoRA precision), and
`fp32_offload` (actual FP32 weights and compute, without BF16 autocast).
The preview page exposes all three choices and can rerun an uploaded video
without replacing its original results. The CLI accepts `--precision` as well.

Precision is a property of each model session, not a global environment toggle.
The single worker closes and frees the previous session before switching modes.
Queued/failed/completed job receipts, prediction model versions, job summaries,
and health responses identify precision. Completed jobs report model-loading,
segmentation and total processing seconds separately.

To avoid repeated offload reads through the OneDrive/Windows mount, copied the
existing caches to WSL's Linux filesystem; the original caches were retained:

```bash
mkdir -p /home/huang/.cache/flora-api
cp -a .runtime/huggingface /home/huang/.cache/flora-api/
cp -a .runtime/models /home/huang/.cache/flora-api/
```

The start script now defaults to these Linux cache locations. Disk offload uses
`~/.cache/flora-api/offload/<precision>/`; each precision has separate files.
No WSL memory, swap, Windows driver or firewall settings were changed.

Ran the complete regression suite: **7 passed**, including precision validation,
NF4→BF16→FP32 session switching, precision-specific prediction versions, and
preservation of the original job during reruns. The switching test uses mock
engines; actual GPU timings are recorded separately.

Verified the new selector in the browser and submitted the existing sample
through **Unquantized · BF16 + FP32 (NVIDIA precision)** → **Run this video again**.
New benchmark job: `3429ddf5cf044815bf69c7726850dcc3`.

That first unquantized attempt hit the reference loader's 300-second generation
limit: 176 tokens generated in 301.846 seconds, without a completed response.
It failed validation and did not export partial predictions. This was not a
completed two-minute prelabeling run. Both the adapter and GDN checkpoint-value
checks passed before inference.

For offline prelabeling, increased per-window generation limits to 300 seconds
for NF4, 900 seconds for native BF16, and 1800 seconds for FP32. Added a bounded
`generation_timeout_s` request override (30–3600 seconds) and CLI
`--generation-timeout`. Increased the default CPU placement budget from 10 GiB
to 12 GiB to reduce disk offloading, without changing WSL's own memory limit.
Ran all seven tests again successfully after these changes.

At the user's direction to use offline jobs, submitted a fresh sample rerun:

```powershell
Invoke-RestMethod -Method Post http://localhost:8008/jobs/a951aa5ccc9a427d836ad44dca036be0/rerun `
  -ContentType 'application/json' `
  -Body '{"precision":"bf16_offload","generation_timeout_s":900}'
```

Offline job ID: `41bbb5744d5b4412847d51900de93bd7`.
Receipt: `.runtime/offline-fullprecision-job.json` (local only).
The job runs independently of the preview browser; keep the API process running.
Final completion time is not yet known. The original five NF4 clips are retained.
FP32 selection/model-switch routing is covered by tests using mock engines; an
actual all-FP32 GPU job has not been validated on this PC.

## INT8 follow-up

Added `precision=int8` to uploads, reruns, the CLI, health metadata and preview
selector. Uses the already-installed bitsandbytes 0.50.2 LLM.int8 implementation
with default outlier threshold 6.0. Quantizes base linear layers and the output
head, keeps the vision encoder BF16 and verifies saved FP32 GDN/LoRA values.
Quantizing the output head, which NF4 leaves unquantized, reserves more GPU
space for INT8 activations. INT8 requests use GPU placement and never silently
switch to NF4 or offloading on failure. Default timeout: 300 seconds per window.

The API queue was idle before restarting to enable INT8. Previous NF4 jobs
and clips were preserved. All **7 regression tests passed**, including routing
and session switching through NF4 → INT8 → BF16 → FP32 with mock engines.

Submitted the existing 24-second sample through the real endpoint:

```powershell
Invoke-RestMethod -Method Post http://localhost:8008/jobs/43315db37d0c47729b9ca483f71294ff/rerun `
  -ContentType 'application/json' -Body '{"precision":"int8"}'
```

INT8 job: `093a88ecc88b49bea91b1a1d91dae871`.
Local receipt: `.runtime/int8-job.json`.
The existing NF4 baseline reported 48.33 seconds total, including 9.09 seconds
model loading, 27.69 seconds segmentation, and four inference windows.

**INT8 completed successfully**: 176.53 seconds total, 68.23 seconds loading,
101.32 seconds segmentation, four windows and five clips. Peak PyTorch CUDA
tensor allocation was **13.522 GiB**. Windows GPU usage was close to capacity
during inference (approximately 15.5 GiB including other allocations), so
headroom is limited. The configuration places all model parameters on CUDA;
there is no configured CPU offload for INT8. This validates this particular
sample, not arbitrary longer contexts or other GPU memory usage.

## Python LAN client follow-up

Created `examples/lan_client.py`, `examples/requirements.txt` (HTTPX only), and
`examples/README.md` for another computer on the same router/LAN. The client
can check health, upload a local video, choose precision, poll/resume a job, and
download result JSON, Label Studio JSON and generated clips. Tokens are read
from `FLORA_API_TOKEN`; HTTP environment proxies are disabled for the selected
local server. Download URLs and job IDs are validated before filesystem writes.

Observed the server's default-gateway interface is Ethernet 2, LAN IPv4
`10.0.0.141`; its Wi-Fi interface has a link-local `169.254.*` address and is
not the endpoint to use. Ethernet on the server and Wi-Fi on the client are
compatible when both devices can communicate on the same LAN.

Verified Windows currently listens only at `127.0.0.1:8008`. The guide provides
optional administrator commands for a Windows port proxy from
`10.0.0.141:8009` to `127.0.0.1:8008`, plus a Private-profile/LocalSubnet firewall
rule. Those changes were documented, **not applied**. No LAN/public listener,
firewall exception or token was created automatically. The Python example
defaults to the documented bridge URL and supports a `--server` override.

Tested the Python client's actual upload → poll → prediction export → clip
download flow against the existing localhost API, using NF4 at the user's
request:

```bash
.runtime/venv/bin/python examples/lan_client.py --server http://127.0.0.1:8008 \
  --video /mnt/c/Users/huang/Downloads/0ac0a71f-98e3-4b09-3273-6d19ede20a7d.mp4 \
  --precision nf4 --output-dir .runtime/nf4-client-test
```

New NF4 job: `095040fc5f384029a435ec59f250f682`, completed with five clips and
four inference windows. Measured 42.53 seconds total, 7.59 seconds loading,
29.07 seconds segmentation, and 11.147 GiB peak CUDA tensor allocation.
The client saved `receipt.json`, `result.json`, `label-studio.json` and all five
clips under its job folder. SHA-256 hashes of every downloaded clip matched
the server's corresponding file. Tested `--job-id` resume: it retrieved the
existing results and preserved the existing clips and original receipt,
without scheduling another inference job.

Client upload/download testing was performed on the same PC over loopback.
Cross-device LAN connectivity remains untested and requires the administrator
bridge/firewall setup in `examples/README.md` and a request from the second PC.

## LAN access enabled after user approval

The second computer reported 100% ping packet loss. The server's Ethernet 2
address was confirmed as `10.0.0.141/24`, with a Public network profile. Ping
failure alone does not establish whether the HTTP API is reachable.

Added `scripts/configure_lan.ps1` with administrator setup and `-Disable` undo.
Automatic approval review initially rejected the persistent networking change
because explicit authorization was missing. After the user said "allow it",
the same administrator command completed with exit code 0.

Applied a persistent Windows port proxy from `10.0.0.141:8009` to
`127.0.0.1:8008`, and firewall rule `FLORA-LAN-8009`, restricted to TCP 8009,
local address `10.0.0.141`, interface Ethernet 2, and clients `10.0.0.0/24`.
The rule applies to all profiles; the Public profile remains unchanged.
No ping rule or router port forwarding was added.

Verified the proxy table, enabled firewall rule, successful setup receipt, and
`http://10.0.0.141:8009/health` from the server PC: status up, CUDA available,
NF4 loaded, queue empty. This confirms the local proxy path. A request from the
second computer is still required to verify cross-device connectivity:

```text
python examples/lan_client.py --server http://10.0.0.141:8009 --check
```

The current LAN setup supersedes the earlier "not applied" networking notes.
Reproduction and undo instructions are in `examples/README.md`.

## Sources

- [NVIDIA model card](https://huggingface.co/nvidia/Flora-TempSeg)
- [NVIDIA inference instructions](https://huggingface.co/nvidia/Flora-TempSeg/blob/b3f079b160a23f78026b72b80fb7a0e05c57eba8/batch_utils/README.md)
- [NVIDIA noncommercial model license](https://huggingface.co/nvidia/Flora-TempSeg/blob/main/LICENSE.pdf)
- [Label Studio TimelineLabels](https://labelstud.io/tags/timelinelabels)
- [HumanSignal frame conversion implementation](https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/examples/yolo/utils/converter.py)

## Database repair after deleting all tasks

The user removed all local task files, including .runtime/jobs/jobs.sqlite3.
The running API then returned HTTP 500 for GET /jobs because the jobs table
was missing. Recreated an empty jobs table; SQLite integrity_check returned
ok and the task count was zero. The loaded NF4 model was kept running.
Database connections now create the data directory and schema when missing.
Added a regression test deleting the database and directory during service
operation, then verifying GET /jobs recovers with an empty list.

