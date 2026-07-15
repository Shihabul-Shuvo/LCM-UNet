"""Run configuration: a frozen, hashable, YAML-serialisable spec for one experiment.

Every experiment (baseline, GLGF, LC-SS2D hero, or any ablation row in
docs/LCM-UNet_FINAL_methodology_v4.1.md §10) is fully described by a RunConfig.
Ablations are config toggles on `model_cfg`, not new model classes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml

# Open dict of LC-SS2D / ablation toggles. Keys are "open" (extensible) by
# design so new ablation rows (e.g. Ablation B's condition target, the Bound
# ablation) can be added later without changing the RunConfig schema.
DEFAULT_MODEL_CFG: Dict[str, Any] = {
    "descriptor_type": "contrast",  # "contrast" (DWConv3x3(Xn)-Xn, default) | "plain" (DWConv3x3(Xn))
    "kernel_size": 3,               # 3 (proposed) | 1 (Ablation C capacity-matched control)
    "inject_target": "delta",       # "delta" (proposed) | "input" (Ablation D)
    "placement": "hero",            # "P0" | "E4E5" | "hero" | "+E6" | "E4D4"
    "use_e6": False,                # LC-VSS at bottleneck E6 (ablation only; default off per §4)
    "alpha_init": 0.01,             # learnable scalar init (§3.3)
    "wdelta_std": 1e-3,             # W_delta init std (§3.4/§5.4)
}

VALID_SCAN_IMPL = ("cuda", "ref")

_ID_LENGTH = 12  # short sha256 hex prefix


@dataclasses.dataclass(frozen=True)
class RunConfig:
    run_name: str
    model_name: str  # e.g. "ultralight_baseline" | "glgf" | "lcm_unet"
    dataset: str      # e.g. "kvasir_seg" | "cvc_clinicdb" | "isic2017" | "isic2018"
    seed: int
    split_file: str

    epochs: int = 250
    batch_size: int = 8
    grad_accum_steps: int = 1
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-2
    input_size: int = 256
    amp: bool = True

    # Same-scan-implementation rule (§5.5): must be identical across every
    # model compared in a given efficiency/accuracy table.
    scan_impl: str = "ref"  # "cuda" | "ref"

    model_cfg: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_MODEL_CFG)
    )

    notes: str = ""

    def __post_init__(self) -> None:
        if self.scan_impl not in VALID_SCAN_IMPL:
            raise ValueError(
                f"scan_impl must be one of {VALID_SCAN_IMPL}, got {self.scan_impl!r}"
            )
        # Fill any missing keys with defaults; keep extra/open keys as given.
        merged = dict(DEFAULT_MODEL_CFG)
        merged.update(self.model_cfg)
        object.__setattr__(self, "model_cfg", merged)

    # -- (de)serialisation -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=True)
        return path

    @classmethod
    def load_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    # -- identity ------------------------------------------------------

    @property
    def config_id(self) -> str:
        """Deterministic short id: sha256 of canonical (sorted-key) JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest[:_ID_LENGTH]
