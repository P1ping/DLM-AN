"""Resolve the matching checkpoint bundle for each DLM-AN inference backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "Piping/DLM-AN"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CheckpointBundle:
    lm_checkpoint: Path
    quantizer_checkpoint: Path
    hift_checkpoint: Path
    flow_checkpoint: Path | None
    lm_config: Path


# The filenames in the model repository must match these paths. HiFT is shared.
PRESETS = {
    "km-flow": {
        "lm_checkpoint": "km-flow/lm.pt",
        "quantizer_checkpoint": "km-flow/quantizer.pt",
        "flow_checkpoint": "km-flow/flow.pt",
        "hift_checkpoint": "shared/hift.pt",
        "lm_config": ROOT / "configs/dlm.paper.model-only.yaml",
    },
    "joint-ctx": {
        "lm_checkpoint": "joint-ctx/lm.pt",
        "quantizer_checkpoint": "joint-ctx/quantizer.pt",
        "flow_checkpoint": None,
        "hift_checkpoint": "shared/hift.pt",
        "lm_config": ROOT / "configs/dlm.model-only.yaml",
    },
}


def resolve_checkpoints(
    backend: str,
    *,
    lm_checkpoint: str | Path | None = None,
    quantizer_checkpoint: str | Path | None = None,
    flow_checkpoint: str | Path | None = None,
    hift_checkpoint: str | Path | None = None,
    lm_config: str | Path | None = None,
    checkpoint_repo: str = DEFAULT_REPO,
) -> CheckpointBundle:
    """Download only missing checkpoint paths; explicit local paths always win."""
    if backend not in PRESETS:
        raise ValueError(f"Unknown backend: {backend}")
    if backend == "joint-ctx" and flow_checkpoint is not None:
        raise ValueError("joint-ctx uses flow weights in its quantizer checkpoint")

    preset = PRESETS[backend]
    provided = {
        "lm_checkpoint": lm_checkpoint,
        "quantizer_checkpoint": quantizer_checkpoint,
        "flow_checkpoint": flow_checkpoint,
        "hift_checkpoint": hift_checkpoint,
    }

    def checkpoint_path(name: str) -> Path | None:
        if provided[name] is not None:
            path = Path(provided[name]).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"{name} does not exist: {path}")
            return path
        filename = preset[name]
        if filename is None:
            return None
        try:
            return Path(hf_hub_download(repo_id=checkpoint_repo, filename=filename))
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {filename} from {checkpoint_repo}; "
                f"supply --{name} to use a local checkpoint"
            ) from exc

    return CheckpointBundle(
        lm_checkpoint=checkpoint_path("lm_checkpoint"),
        quantizer_checkpoint=checkpoint_path("quantizer_checkpoint"),
        flow_checkpoint=checkpoint_path("flow_checkpoint"),
        hift_checkpoint=checkpoint_path("hift_checkpoint"),
        lm_config=(
            Path(lm_config).expanduser()
            if lm_config is not None
            else preset["lm_config"]
        ),
    )
