"""Small coverage head and checkpoint IO, independent of raster and CROMA code."""

from pathlib import Path

import torch
from torch import nn


class CoverageHead(nn.Module):
    """Frozen 768-value feature -> 19 logits; softmax gives predicted coverage."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 19)

    def forward(self, features):
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim < 2
            or features.shape[-1] != 768
            or features.numel() == 0
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Features must be nonempty finite float32 (...,768) tensors")
        return self.linear(features)

    @torch.inference_mode()
    def predict(self, features):
        self.eval()
        return self(features).softmax(dim=-1)


def coverage_loss(logits, fractions):
    """Soft-target cross entropy, with the full coverage distribution retained."""
    if (
        not isinstance(logits, torch.Tensor)
        or not isinstance(fractions, torch.Tensor)
        or logits.ndim < 2
        or logits.shape[-1] != 19
        or logits.numel() == 0
        or logits.shape != fractions.shape
        or fractions.dtype != torch.float32
        or not torch.isfinite(logits).all()
        or not torch.isfinite(fractions).all()
        or (fractions < 0).any()
        or (fractions > 1).any()
        or not torch.allclose(
            fractions.sum(-1), torch.ones_like(fractions[..., 0]), atol=1e-6, rtol=0
        )
    ):
        raise ValueError("Expected finite logits and 19 nonnegative fractions summing to one")
    return -(fractions * logits.log_softmax(dim=-1)).sum(dim=-1).mean()


def save_head(path: Path, model: CoverageHead, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing trained weights.
    with path.open("xb") as stream:
        torch.save(
            {
                "format_version": 1,
                "architecture": "linear_768_19",
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata": metadata,
            },
            stream,
        )


def load_head(path: Path) -> tuple[CoverageHead, dict]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, dict)
        or value.get("format_version") != 1
        or value.get("architecture") != "linear_768_19"
        or not isinstance(value.get("metadata"), dict)
    ):
        raise ValueError("Expected a linear_768_19 checkpoint version 1")
    with torch.random.fork_rng(devices=[]):
        model = CoverageHead()
    model.load_state_dict(value["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Checkpoint contains nonfinite weights")
    return model.eval(), value["metadata"]
