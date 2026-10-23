"""Real GPU inference, using a synthetic video to check execution, not quality."""
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
from flora_api.engine import Engine

path = Path(".runtime/smoke.mp4")
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",
                "testsrc2=duration=4:size=320x240:rate=24", "-c:v", "libx264", str(path)], check=True)
result = Engine().segment(path, "human", "")
Path(".runtime/smoke-result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
