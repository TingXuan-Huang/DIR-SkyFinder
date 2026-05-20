"""Camera-conditioned variant of the baseline model.

Mechanism: a learnable per-camera embedding is *added* to the image-backbone
feature before the regression head. Test-time cameras (not in train) get a
special "unknown" embedding that is trained alongside the others via 5% random
camera dropout — this gives the model a sensible default for LOCO test rows
without leaking camera identity.

This is the upper-bound experiment for Analysis #7 in REPORT.md §11.

Run via the standard YAML pipeline:

    # config.yaml row
    experiments:
      - run_name: cam_cond_baseline_resnet50_fold0
        model: resnet50
        fold: 0
        use_lds: false
        use_fds: false
        # ↓ new flags this module recognises
        use_cam_embedding: true
        cam_embedding_dim: 64
        cam_dropout_prob: 0.05

The runner needs a 6-line patch (see PATCH_RUNNER below). Without that patch,
`make_cam_conditioned_model` can be called directly from a script.

Quick smoke test (CPU, no images needed):
    python -m dir_skyfinder.cam_conditioned --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from skyfinder.training.model import build_model as make_model

UNKNOWN_CAM_TOKEN = "__unk__"


class CamConditionedModel(nn.Module):
    """Wraps `make_model(...)` and adds a camera-id embedding.

    The image backbone produces `f_img` (B, D). A learnable embedding
    `e_cam` (B, E) is projected to D and added: `f = f_img + W e_cam`.
    The shared regression head reads `f`. With cam_dropout_prob > 0, a
    fraction of training cameras are randomly replaced with the unknown
    token so the unknown embedding sees real gradients.
    """

    def __init__(self, backbone_name: str,
                 cam_id_to_idx: dict[int, int],
                 emb_dim: int = 64,
                 cam_dropout_prob: float = 0.05,
                 freeze_backbone: bool = False):
        super().__init__()
        base = make_model(backbone_name, freeze_backbone=freeze_backbone)

        # Pop the regression head off so we get raw features.
        if hasattr(base, "fc") and isinstance(base.fc, nn.Linear):
            self.feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        elif hasattr(base, "heads"):
            self.feat_dim = base.heads.head.in_features
            base.heads.head = nn.Identity()
        else:
            raise ValueError(f"unsupported backbone: {backbone_name}")
        self.backbone = base

        # +1 for the unknown-camera token, kept at index 0.
        self.cam_id_to_idx = dict(cam_id_to_idx)
        self.n_cams = max(cam_id_to_idx.values()) + 1
        self.unknown_idx = 0
        assert self.unknown_idx not in cam_id_to_idx.values(), \
            "cam_id_to_idx must reserve idx 0 for the unknown token"
        self.cam_emb = nn.Embedding(self.n_cams, emb_dim)
        self.cam_proj = nn.Linear(emb_dim, self.feat_dim)
        # Make the head a fresh single-output layer.
        self.head = nn.Linear(self.feat_dim, 1)
        self.cam_dropout_prob = cam_dropout_prob

    def cam_idx(self, cam_ids: torch.Tensor) -> torch.Tensor:
        """Map raw CamId tensor → embedding index. Unknown cams → 0."""
        out = torch.zeros_like(cam_ids, dtype=torch.long)
        for i, c in enumerate(cam_ids.tolist()):
            out[i] = self.cam_id_to_idx.get(int(c), self.unknown_idx)
        return out

    def forward(self, x: torch.Tensor, cam_ids: torch.Tensor) -> torch.Tensor:
        f_img = self.backbone(x)
        if f_img.ndim > 2:
            f_img = f_img.flatten(1)
        cam_idx = self.cam_idx(cam_ids).to(x.device)
        # In training, randomly replace some cam idxs with the unknown token
        # so the unknown embedding gets gradient. This is the mechanism that
        # makes LOCO test feasible.
        if self.training and self.cam_dropout_prob > 0:
            mask = torch.rand_like(cam_idx, dtype=torch.float32) < self.cam_dropout_prob
            cam_idx = torch.where(mask, torch.full_like(cam_idx, self.unknown_idx),
                                  cam_idx)
        e = self.cam_proj(self.cam_emb(cam_idx))
        return self.head(f_img + e).squeeze(-1)


def build_cam_id_to_idx(train_cam_ids: list[int]) -> dict[int, int]:
    """Assign each train camera an index >=1; reserves 0 for unknown."""
    return {int(c): i + 1 for i, c in enumerate(sorted(set(train_cam_ids)))}


# --- Patch needed in the existing trainer ---------------------------------

PATCH_RUNNER = """\
# Inside dir_skyfinder/baseline.py (or whichever module owns the training
# loop), make these surgical additions to support cam-conditioning:

# 1. SkyFinderDataset.__getitem__ should also return CamId (or wrap it):
#       return x, y, w, torch.tensor(int(row["CamId"]), dtype=torch.long)
#    Backward-compat: in the existing loader, unpack as (x, y, w, *rest).

# 2. In your train step, if `cfg.use_cam_embedding`:
#       cam_ids = batch[3].to(device)
#       preds = model(x, cam_ids)
#    Otherwise call model(x) as before.

# 3. Build the model:
#       if cfg.use_cam_embedding:
#           cam_id_to_idx = build_cam_id_to_idx(train_df["CamId"].tolist())
#           model = CamConditionedModel(
#               cfg.model, cam_id_to_idx,
#               emb_dim=cfg.cam_embedding_dim,
#               cam_dropout_prob=cfg.cam_dropout_prob,
#               freeze_backbone=cfg.freeze_backbone)
#       else:
#           model = make_model(cfg.model, freeze_backbone=cfg.freeze_backbone)

# 4. At inference time on LOCO test cameras, the unknown token kicks in
#    automatically — no code change needed, just pass real CamIds.

# Expected outcome on fold-0 ResNet-50:
#   - val MAE should NOT degrade vs baseline (~2.8 °C)
#   - test MAE should drop, but by HOW MUCH is the experiment.
#     If it drops to <= C2 (~6.4 °C), the camera prior is the missing signal.
#     If it stays at ~7.3 °C, image content really is the bottleneck.
"""


# --- Smoke test ----------------------------------------------------------

def _smoke() -> None:
    cam_ids = [10, 11, 12, 13, 14, 15, 99]  # 99 unseen
    idx = build_cam_id_to_idx([10, 11, 12, 13, 14, 15])
    print("cam_id_to_idx:", idx)

    m = CamConditionedModel("resnet50", idx, emb_dim=8, cam_dropout_prob=0.5)
    m.train()
    x = torch.randn(len(cam_ids), 3, 224, 224)
    cam_t = torch.tensor(cam_ids, dtype=torch.long)
    out = m(x, cam_t)
    print("train out shape:", out.shape, "all finite:", bool(torch.isfinite(out).all()))

    m.eval()
    with torch.no_grad():
        out = m(x, cam_t)
        # Unseen cam (99) should fall through to the unknown token deterministically.
        out_unk = out[-1].item()
        out_unk2 = m(x[-1:], cam_t[-1:]).item()
        print("unseen-cam pred is deterministic:", abs(out_unk - out_unk2) < 1e-5)
    print("smoke OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
    else:
        parser.print_help()
        print()
        print(PATCH_RUNNER)
