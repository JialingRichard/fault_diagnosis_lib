"""
TiViT2one (framework-native)
=============================

Minimal integration of TiViT for sequence-to-one classification using
HuggingFace backbones (no open_clip dependency in this first version).

Pipeline per channel:
  (B, T, 1) -> ts->image (grayscale, resize to 224) -> ViT (frozen) ->
  hidden aggregation ('mean' or 'cls_token') -> (B, D_hidden)
Concatenate over input_dim channels -> (B, input_dim * D_hidden) -> Linear -> (B, num_classes)

Defaults:
- vit_name: 'facebook/dinov2-base'
- vit_layer: 14 (sliced/truncated; safe if > actual depth)
- aggregation: 'mean'
- patch_size: 'sqrt' (computed as round(sqrt(T)))
- stride: 1.0 (no overlap)
- image_size: 224
- freeze_vit: True
- l2norm: True

Note: This file intentionally avoids sklearn/aeon/open_clip to keep dependencies minimal.
"""

from typing import Optional, Union, List
import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T  # kept for potential future use

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    ViTMAEForPreTraining,
)


def _compute_patch_size(policy: Union[str, int], T: int, min_ps: int = 16) -> int:
    """Resolve patch size from policy or numeric value.

    Args:
        policy: 'sqrt'|'auto' or an integer
        T: time steps
        min_ps: lower bound for patch size
    """
    if isinstance(policy, str):
        pol = policy.strip().lower()
        if pol in {"sqrt", "auto"}:
            return max(1, int(round(math.sqrt(max(1, T)))))
        else:
            raise ValueError(f"Unsupported patch_size policy: {policy}")
    try:
        return max(1, int(policy))
    except Exception:
        raise ValueError(f"Invalid patch_size value: {policy}")


def _get_processor_vit_hf(model_name: str):
    """Create HF processor and ViT backbone for supported families.

    Supports CLIP (HF), DINOv2, SigLIP2, MAE. Does not include open_clip.
    Returns (processor, vit_module)
    """
    name = model_name.lower()
    if "clip" in name:
        # Use HF CLIP (vision part)
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        vit = model.vision_model
    elif "dinov2" in name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        vit = AutoModel.from_pretrained(model_name)
    elif "siglip" in name:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        vit = model.vision_model
    elif "mae" in name:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTMAEForPreTraining.from_pretrained(model_name)
        vit = model.vit
    else:
        raise ValueError(f"Unsupported model for HF route: {model_name}")
    return processor, vit


class BaseTiViT(nn.Module):
    """Base class that converts TS -> Image and aggregates hidden states."""

    def __init__(self, processor, vit, layer_idx: Optional[int], aggregation: str, patch_size: int, stride: float, image_size: int = 224):
        super().__init__()
        self.processor = processor
        self.vit = vit
        self.layer_idx = layer_idx
        self.aggregation = str(aggregation).lower()
        self.patch_size = int(patch_size)
        self.stride = float(stride)
        self.image_size = int(image_size)
        self.truncate_layers()

    def truncate_layers(self):
        """Override in subclass if truncation supported."""
        return

    def forward_vit(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run ViT forward and return token representations (B, tokens, D)."""
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, T, 1)
        imgs = self.ts2image_transformation(inputs, patch_size=self.patch_size, stride=self.stride, image_size=self.image_size)
        hidden = self.forward_vit(imgs)
        return self.aggregate_hidden_representations(hidden, aggregation=self.aggregation)

    @staticmethod
    def aggregate_hidden_representations(hidden_states: torch.Tensor, aggregation: str) -> torch.Tensor:
        if aggregation == "mean":
            pooled = hidden_states.mean(dim=1)
        elif aggregation == "cls_token":
            pooled = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        return pooled

    @staticmethod
    def ts2image_transformation(x: torch.Tensor, patch_size: int, stride: float, image_size: int = 224) -> torch.Tensor:
        """Convert 1D time series (B, T, 1) to grayscale image tensor batch (B, 3, H, W).

        Steps:
        - Robust scaling by median/IQR along time dim
        - Segment time into windows of length patch_size with optional overlap (stride in (0,1])
        - Contrast adjust (min-max to [0,1], gamma=0.8)
        - Resize to (image_size, image_size) and replicate to 3 channels
        """
        # Expect x: (B, T, 1)
        if x.dim() != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected input (B, T, 1), got {tuple(x.shape)}")

        # Robust scaling
        median = x.median(1, keepdim=True)[0]
        q_tensor = torch.tensor([0.75, 0.25], device=x.device, dtype=x.dtype)
        q75, q25 = torch.quantile(x, q_tensor, dim=1, keepdim=True)
        x = x - median
        iqr = q75 - q25
        x = x / (iqr + 1e-5)

        # (B, 1, T)
        x = einops.rearrange(x, "b t d -> b d t")
        Tlen = x.shape[-1]

        if stride == 1 or stride == 1.0:
            # No overlap; pad left to be divisible by patch_size
            pad_left = 0
            if Tlen % patch_size != 0:
                pad_left = patch_size - (Tlen % patch_size)
            x_pad = F.pad(x, (pad_left, 0), mode="replicate")
            # (B, 1, T+pad) -> (B*1, 1, f, p)
            x_2d = einops.rearrange(x_pad, "b d (p f) -> (b d) 1 f p", f=patch_size)
        elif 0 < stride < 1:
            stride_len = int(patch_size * stride)
            if stride_len <= 0:
                stride_len = 1
            remainder = (Tlen - patch_size) % stride_len
            pad_left = 0
            if remainder != 0:
                pad_left = stride_len - remainder
            x_pad = F.pad(x, (pad_left, 0), mode="replicate")
            # unfold over time: (B, 1, T') -> (B, 1, f, p)
            x_2d = x_pad.unfold(dimension=2, size=patch_size, step=stride_len)
        else:
            raise ValueError(f"Stride must be 1 or in (0,1): got {stride}")

        # Contrast adjust per window
        min_vals = x_2d.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_vals = x_2d.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x_2d = (x_2d - min_vals) / (max_vals - min_vals + 1e-5)
        x_2d = torch.pow(x_2d, 0.8)

        # Resize to ViT input resolution (batched 4D tensor) using tensor op
        x_resized = F.interpolate(x_2d, size=(image_size, image_size), mode="bilinear", align_corners=False)

        # Generate 3-channel images
        image_input = einops.repeat(x_resized, "b 1 h w -> b c h w", c=3)
        return image_input


class TiViT_HF(BaseTiViT):
    """HF-backed TiViT with PIL-based processors."""

    def __init__(self, processor, vit, layer_idx: Optional[int], aggregation: str, patch_size: int, stride: float, image_size: int = 224):
        super().__init__(processor, vit, layer_idx, aggregation, patch_size, stride, image_size)
        # No PIL/processor path; operate on tensors directly

    def truncate_layers(self):
        # slice encoder layers up to layer_idx (if provided)
        if self.layer_idx is None or self.layer_idx == -1:
            return
        try:
            if hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layers"):
                self.vit.encoder.layers = self.vit.encoder.layers[: self.layer_idx]
            elif hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer"):
                self.vit.encoder.layer = self.vit.encoder.layer[: self.layer_idx]
        except Exception:
            # best-effort; if architecture differs, skip truncation
            pass

    def forward_vit(self, inputs: torch.Tensor) -> torch.Tensor:
        # Tensor-native normalization and forward
        # inputs: (B, 3, H, W) float in [0,1]
        device = inputs.device
        dtype = inputs.dtype

        mean, std = self._get_norm_tensors(device=device, dtype=dtype)
        pixel_values = (inputs - mean) / std

        # HF models expect keyword 'pixel_values'
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=(self.layer_idx is None))
        if self.layer_idx is not None:
            # last_hidden_state after truncation
            return outputs.last_hidden_state
        else:
            # stack all hidden states if requested (not used in this minimal integration)
            return torch.stack(outputs.hidden_states, dim=-1)

    def _get_norm_tensors(self, device: torch.device, dtype: torch.dtype):
        name = str(self.vit_name).lower() if hasattr(self, 'vit_name') else ""
        # Default to ImageNet stats
        if ("clip" in name) or ("siglip" in name):
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype)
        else:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        return mean, std


class TiViT2one(nn.Module):
    """
    Framework-native classifier that wraps a frozen HF ViT via TiViT pipeline.

    Expected input: (batch, seq_len, input_dim)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        vit_name: str = "facebook/dinov2-base",
        vit_layer: Optional[int] = 14,
        aggregation: str = "mean",
        patch_size: Union[str, int] = "sqrt",
        stride: float = 1.0,
        image_size: int = 224,
        freeze_vit: bool = True,
        l2norm: bool = True,
        head_dropout: float = 0.0,
        time_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.vit_name = vit_name
        self.vit_layer = vit_layer
        self.aggregation = str(aggregation).lower()
        self.patch_size_policy = patch_size  # can be 'sqrt' or int
        self.stride = float(stride)
        self.image_size = int(image_size)
        self.freeze_vit = bool(freeze_vit)
        self.l2norm = bool(l2norm)
        self.head_dropout = float(head_dropout)
        self.time_steps = int(time_steps) if time_steps is not None else None

        # Build HF processor and vit
        processor, vit = _get_processor_vit_hf(self.vit_name)
        # Temporary patch size placeholder (will be updated per forward if 'sqrt')
        init_ps = 1 if isinstance(self.patch_size_policy, str) else int(self.patch_size_policy)
        self.backbone = TiViT_HF(
            processor=processor,
            vit=vit,
            layer_idx=self.vit_layer,
            aggregation=self.aggregation,
            patch_size=init_ps,
            stride=self.stride,
            image_size=self.image_size,
        )
        # expose vit_name to backbone for normalization decision
        setattr(self.backbone, 'vit_name', self.vit_name)

        # Freeze ViT if requested
        if self.freeze_vit:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # Infer hidden size from backbone config (robust across families)
        hidden_size = self._infer_hidden_size(self.backbone.vit)
        embed_dim = self.input_dim * hidden_size

        layers: List[nn.Module] = []
        if self.head_dropout and self.head_dropout > 0:
            layers.append(nn.Dropout(self.head_dropout))
        layers.append(nn.Linear(embed_dim, self.output_dim))
        self.classifier = nn.Sequential(*layers)

    @staticmethod
    def _infer_hidden_size(vit: nn.Module) -> int:
        # Prefer config.hidden_size
        cfg = getattr(vit, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        # Fallbacks (rare)
        for attr in ("hidden_size", "embed_dim", "width"):
            val = getattr(vit, attr, None)
            if isinstance(val, int):
                return int(val)
        raise RuntimeError("Cannot infer ViT hidden size from backbone")

    def _resolve_patch_size(self, T: int) -> int:
        return _compute_patch_size(self.patch_size_policy, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        if x.dim() != 3 or x.shape[2] != self.input_dim:
            raise ValueError(f"Expected (B, T, C={self.input_dim}), got {tuple(x.shape)}")

        B, T, C = x.shape
        # Resolve patch size per input T
        ps = self._resolve_patch_size(T)
        # Update backbone patch size for this forward
        self.backbone.patch_size = ps

        # Channel-as-batch: (B, T, C) -> (B*C, T, 1)
        x_bc = einops.rearrange(x, "b t c -> (b c) t 1")
        if self.freeze_vit:
            with torch.no_grad():
                feats_flat = self.backbone(x_bc)  # (B*C, D)
        else:
            feats_flat = self.backbone(x_bc)
        # reshape back: (B, C*D)
        z = einops.rearrange(feats_flat, "(b c) d -> b (c d)", b=B, c=C)
        if self.l2norm:
            z = F.normalize(z, p=2, dim=-1)
        logits = self.classifier(z)
        return logits
