# Python client from another computer on the same LAN

The GPU server is connected by **Ethernet** at `10.0.0.141`. A client connected
over Wi-Fi works when both devices can communicate through the same router/LAN.
The server's Wi-Fi adapter's `169.254.*` address is not the address to use.
No GPU, CUDA, FLORA model or Label Studio installation is needed on the client.

## One-time server networking setup

The API currently listens through Windows at **127.0.0.1:8008**. That address is
only accessible on the server PC. The following optional Windows bridge exposes
it at **http://10.0.0.141:8009** without restarting the API or reloading the GPU.
This bridge was enabled on this PC with the user's approval. The Python client
does not change networking settings.

To reproduce the setup, keep the API running and open **PowerShell as
Administrator** in the repository root:

```powershell
powershell.exe -NoProfile -File .\scripts\configure_lan.ps1
```

The helper creates a persistent port proxy and firewall rule `FLORA-LAN-8009`.
The rule permits TCP port 8009 on `10.0.0.141`, on **Ethernet 2** only, from
clients in **10.0.0.0/24**. It applies to all network profiles so the existing
Public Ethernet profile can remain unchanged. It does not disable the firewall
or enable ping. Parameters allow adapting the address, interface, subnet and
ports to another installation. Setup results are saved in
`.runtime/lan-setup-result.json`.

Keep the existing WSL API process running. Windows's IP Helper service handles
the port proxy; check `Get-Service iphlpsvc` if the bridge does not listen.
This forwards the existing Windows localhost listener, rather than directly
exposing WSL's changing `172.*` address.

Optional API authentication: set `FLORA_API_TOKEN` in Ubuntu **before** starting
the API, then set the same token in the client's environment. Do not put it in
the Python file or a URL. Restarting the API interrupts active jobs, so wait
for the queue to become idle before changing its environment.

```bash
# In Ubuntu on the server, before bash scripts/start.sh:
export FLORA_API_TOKEN='your-long-random-token'
bash scripts/start.sh
```

If DHCP changes the server's Ethernet address, update the bridge and the
client's `--server` argument. Guest Wi-Fi/client isolation can prevent devices
from reaching one another even when they share a router.

Undo just this bridge/firewall rule if no longer needed (Administrator):

```powershell
powershell.exe -NoProfile -File .\scripts\configure_lan.ps1 -Disable
```

See [Microsoft's WSL networking guide](https://learn.microsoft.com/en-us/windows/wsl/networking)
for localhost forwarding and `netsh interface portproxy` background.

## Run on the client computer

Copy `lan_client.py` and `requirements.txt` from this folder, or clone the
repository and run these commands from its root. Use Python 3.10 or later.

```powershell
python -m pip install -r examples/requirements.txt
# Only when authentication is enabled on the server:
$env:FLORA_API_TOKEN = "your-long-random-token"

python examples/lan_client.py --server http://10.0.0.141:8009 --check
python examples/lan_client.py --server http://10.0.0.141:8009 --video "C:\videos\demo.mp4" --precision int8
```

On macOS/Linux, use `python3` and `export FLORA_API_TOKEN='...'` instead.
If you copied only these two files to a folder, remove `examples/` from paths.
The video path is on **the client computer**, not on the GPU server.

The example uploads a multipart file to `POST /jobs`, receives HTTP 202, polls
`GET /jobs/JOB_ID`, and downloads the result, Label Studio JSON, and all clips:

```text
flora-results/JOB_ID/
  receipt.json
  result.json
  label-studio.json
  clip_0000.mp4
  ...
```

To resume polling/downloading without submitting another inference job:

```powershell
python examples/lan_client.py --server http://10.0.0.141:8009 --job-id JOB_ID
```

Precision choices: `nf4`, `int8`, `bf16_offload`, `fp32_offload`. The default is
NF4. For slow offline jobs, increase `--wait-timeout` (default 7200 seconds).
`--generation-timeout` controls the server's per-window deadline, separately
from the client's total polling timeout. Closing the client does not cancel
an accepted server job. Existing downloaded clips are preserved on resume.

Label Studio JSON points to server-hosted video by default. If API auth is on,
browser video elements cannot attach the token themselves; use `--video-url`
with your existing accessible media store, or a proxy that handles media auth.
Downloaded clips can also be imported as videos for the next labeling stage.

For a same-PC test without enabling LAN access, use
`--server http://127.0.0.1:8008`. Cross-device connectivity must be checked from
the second computer after enabling the bridge; a same-PC test cannot prove it.
