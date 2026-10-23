"""Download pinned NVIDIA code, not model weights; keep third-party code out of git."""
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "b3f079b160a23f78026b72b80fb7a0e05c57eba8"
if __name__ == "__main__":
    snapshot_download("nvidia/Flora-TempSeg", revision=REVISION,
                      allow_patterns=["batch_utils/*", "LICENSE.pdf", "README.md"],
                      local_dir=ROOT / ".runtime" / "reference")
