"""Temporal-only adapter to NVIDIA's pinned, integrity-checked reference pipeline."""
import math
from bisect import bisect_left
import os
from pathlib import Path


class ModelSession:
    """Experimental 16 GB deployment; retain FP32 GDN/LoRA in both modes."""
    def __init__(self):
        self.model = self.proc = None

    def load(self, kind):
        if kind != "tempseg":
            raise ValueError("This service only loads the segmenter")
        if self.model is not None:
            return self.model, self.proc
        import torch
        from batch_utils.models import snapshot, validate_manifest, processor
        from batch_utils.config import MODELS
        from huggingface_hub import snapshot_download
        from production_gdn_fix import _keep_native_gdn_fp32
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel
        from peft import PeftModel
        from tempseg_fp32_loader import digest
        from .verification import verify_values
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.set_num_threads(4)
        _keep_native_gdn_fp32(Qwen3_5PreTrainedModel)
        adapter = snapshot("tempseg")
        validate_manifest(adapter)
        spec = MODELS["tempseg_base"]
        base = snapshot_download(spec["repo"], revision=spec["revision"],
                                 allow_patterns=["*.safetensors", "*.json", "*.jinja"])
        weights = adapter / "adapter_model.safetensors"
        if digest(weights) != MODELS["tempseg"]["sha256"]:
            raise RuntimeError("Adapter checksum mismatch")
        offload = Path(os.environ.get("FLORA_OFFLOAD_DIR", ".runtime/offload"))
        offload.mkdir(parents=True, exist_ok=True)
        memory = {0: os.environ.get("FLORA_GPU_MEMORY", "10GiB"),
                  "cpu": os.environ.get("FLORA_CPU_MEMORY", "10GiB")}
        mode = os.environ.get("FLORA_PRECISION", "nf4")
        if mode not in {"nf4", "bf16_offload"}:
            raise ValueError("FLORA_PRECISION must be nf4 or bf16_offload")
        placement = {"device_map": "auto", "max_memory": memory,
                     "offload_folder": str(offload)}
        if mode == "nf4":
            placement = {"device_map": {"": 0}, "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])}
        model, info = AutoModelForImageTextToText.from_pretrained(
            base, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True, **placement)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Incomplete base model load: {info}")
        # Transformers 5.8 stores these as sets; Accelerate 1.12's PEFT
        # redispatch path expects lists and otherwise wraps a set in a list.
        for module in model.modules():
            if isinstance(getattr(module, "_no_split_modules", None), set):
                module._no_split_modules = sorted(module._no_split_modules)
        model = PeftModel.from_pretrained(model, adapter, adapter_name="candidate",
                                         autocast_adapter_dtype=True,
                                         device_map={"": 0} if mode == "nf4" else "auto", max_memory=memory,
                                         offload_folder=str(offload)).eval()
        print("FLORA_OFFLOAD_VERIFICATION", verify_values(model, base, weights), flush=True)
        # Match NVIDIA's BF16 compute autocast without converting FP32 parameters.
        import types
        original = model.generate
        def generate(self, *args, **kwargs):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return original(*args, **kwargs)
        model.generate = types.MethodType(generate, model)
        self.model, self.proc = model, processor("tempseg", adapter)
        return self.model, self.proc


class Engine:
    def __init__(self):
        self.session = ModelSession()

    def segment(self, video, profile, instruction):
        from batch_utils.api import DataEnginePipeline
        from batch_utils.pipeline import segment
        from batch_utils.config import PROFILE_POOLS, MODELS
        videos, canonical = DataEnginePipeline._inputs(video, profile, None, None, False, instruction)
        main = videos["main"]
        if not main.cfr:
            raise ValueError("Label Studio timeline export requires a constant-frame-rate video; normalize to CFR first")
        rows, audit = segment(videos, PROFILE_POOLS[canonical], self.session,
                              instruction, lambda *a, **k: None)
        duration = videos["main"].duration_s
        for row in rows:
            if not (math.isfinite(row["start_s"]) and math.isfinite(row["end_s"])
                    and 0 <= row["start_s"] < row["end_s"] <= duration + .001):
                raise RuntimeError("Invalid source interval returned by model")
            pts = [float(t / 1000000) for t in main.pts_us]
            row["start_frame"] = min(len(pts), bisect_left(pts, row["start_s"]) + 1)
            row["end_frame"] = max(row["start_frame"], bisect_left(pts, row["end_s"]))
        return {"duration_s": duration, "segments": rows, "windows": len(audit),
                "fps": float(main.rate), "frame_count": len(main.pts_us),
                "model_revisions": MODELS, "input_sha256": videos["main"].sha256,
                "deployment": "experimental-" + os.environ.get("FLORA_PRECISION", "nf4")}
