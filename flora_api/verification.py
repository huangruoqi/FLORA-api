"""Checkpoint equality checks that also inspect Accelerate's offload backing store."""
import json
from pathlib import Path


def parameter_value(model, name):
    parameter = model.get_parameter(name)
    if parameter.device.type != "meta":
        return parameter.detach().cpu()
    # Accelerate replaces offloaded parameters with meta tensors. Verify the
    # backing values rather than materializing the whole model in host memory.
    parts = name.split(".")
    for i in range(len(parts) - 1, -1, -1):
        module = model.get_submodule(".".join(parts[:i])) if i else model
        hook = getattr(module, "_hf_hook", None)
        for child in getattr(hook, "hooks", [hook]):
            weights = getattr(child, "weights_map", None)
            if weights is not None:
                try:
                    return weights[".".join(parts[i:])].detach().cpu()
                except KeyError:
                    continue
    raise RuntimeError(f"Cannot verify offload backing value for {name}")


def verify_values(model, base, adapter):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    saved_adapter = load_file(str(adapter), device="cpu")
    names = {name.replace(".candidate.", "."): name
             for name, _ in model.named_parameters() if ".candidate." in name}
    if set(names) != set(saved_adapter):
        raise RuntimeError("Adapter parameter names do not match checkpoint")
    for key, saved in saved_adapter.items():
        actual = parameter_value(model, names[key])
        if actual.dtype != torch.float32 or not torch.equal(actual, saved):
            raise RuntimeError(f"FP32 adapter value changed: {key}")
    mapping = json.loads((Path(base) / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {key for key in mapping if key.endswith((".A_log", ".dt_bias"))}
    if not wanted:
        raise RuntimeError("No GDN parameters found")
    for key in sorted(wanted):
        actual = parameter_value(model, "base_model.model." + key)
        with safe_open(str(Path(base) / mapping[key]), framework="pt", device="cpu") as source:
            saved = source.get_tensor(key)
        if actual.dtype != torch.float32 or not torch.equal(actual, saved.float()):
            raise RuntimeError(f"FP32 GDN value changed: {key}")
    return {"adapter_tensors": len(names), "gdn_tensors": len(wanted),
            "all_saved_values_exact": True, "offload_backing_values_checked": True}
