import json
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from flora_api.verification import parameter_value, verify_values


def test_offload_backing_is_checked():
    model = torch.nn.Linear(2, 2, bias=False, device="meta")
    expected = torch.ones(2, 2)
    model._hf_hook = SimpleNamespace(weights_map={"weight": expected})
    assert torch.equal(parameter_value(model, "weight"), expected)
    del model._hf_hook
    with pytest.raises(RuntimeError, match="Cannot verify"):
        parameter_value(model, "weight")


def test_adapter_and_gdn_values_must_remain_exact(tmp_path):
    layer = torch.nn.Module()
    layer.A_log = torch.nn.Parameter(torch.tensor([.123456789], dtype=torch.float32))
    layer.dt_bias = torch.nn.Parameter(torch.tensor([.987654321], dtype=torch.float32))
    layer.lora_A = torch.nn.ModuleDict({"candidate": torch.nn.Linear(1, 1, bias=False)})
    inner = torch.nn.Module()
    inner.layer = layer
    base = torch.nn.Module()
    base.model = inner
    model = torch.nn.Module()
    model.base_model = base
    mapping = {"layer.A_log": "base.safetensors", "layer.dt_bias": "base.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    save_file({"layer.A_log": layer.A_log.detach(), "layer.dt_bias": layer.dt_bias.detach()},
              tmp_path / "base.safetensors")
    adapter = tmp_path / "adapter.safetensors"
    save_file({"base_model.model.layer.lora_A.weight": layer.lora_A.candidate.weight.detach()}, adapter)
    assert verify_values(model, tmp_path, adapter)["all_saved_values_exact"]
    with torch.no_grad():
        layer.A_log.copy_(layer.A_log.bfloat16().float())
    with pytest.raises(RuntimeError, match="GDN value changed"):
        verify_values(model, tmp_path, adapter)
