import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SketchSource:
    """Configuration for loading or computing sketch tokens."""
    mode: str = "precomputed"  # precomputed | external
    field: str = "sketch_path"


class SketchExtractor:
    """Abstract sketch extractor.

    Implementations should return a LongTensor of shape [T].
    """

    def extract(  # pragma: no cover - interface
        self,
        wav: torch.Tensor,
        sample_rate: int,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class PrecomputedSketchExtractor(SketchExtractor):
    """Loads sketch tokens from files referenced in JSONL (sketch_path)."""

    def __init__(self, field: str = "sketch_path"):
        self.field = field

    def extract_from_item(self, item: dict) -> torch.Tensor:
        path = item.get(self.field)
        if not path:
            raise ValueError(f"Missing {self.field} for sketch tokens in JSONL item: {item.get('idx', '<no idx>')}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"sketch_path not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pt":
            data = torch.load(path, map_location="cpu")
            if isinstance(data, dict) and "sketch" in data:
                data = data["sketch"]
            tokens = torch.as_tensor(data, dtype=torch.long)
        elif ext in (".npy", ".npz"):
            data = np.load(path)
            if isinstance(data, np.lib.npyio.NpzFile):
                if "sketch" in data:
                    data = data["sketch"]
                else:
                    raise ValueError(f"NPZ missing 'sketch' array: {path}")
            tokens = torch.from_numpy(np.asarray(data)).long()
        else:
            raise ValueError(f"Unsupported sketch token file extension: {path}")
        if tokens.ndim != 1:
            tokens = tokens.reshape(-1)
        return tokens

    def extract(
        self,
        wav: torch.Tensor,
        sample_rate: int,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Use extract_from_item for precomputed tokens.")


class ExternalSketchExtractor(SketchExtractor):
    """Placeholder for external sketch extraction (e.g., MuQ + VQ).

    Replace this with your MuQ pipeline, or precompute tokens and use
    PrecomputedSketchExtractor instead.
    """

    def __init__(
        self,
        model_id: str = "OpenMuQ/MuQ-large-msd-iter",
        device: str = "cpu",
        sample_rate: int = 24000,
        embedding_dim: int = 1024,
        codebook_size: int = 16384,
        vq_path: Optional[str] = None,
        vq_decay: float = 0.99,
        commitment_weight: float = 1.0,
        freeze_codebook: bool = True,
    ):
        try:
            from muq import MuQ
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "MuQ is not installed. Install it separately and retry "
                "(see https://github.com/tencent-ailab/MuQ)."
            ) from exc

        from vector_quantize_pytorch import VectorQuantize

        self.model = MuQ.from_pretrained(model_id)
        self.model.eval()
        self.model.to(device)

        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size

        self.vq = VectorQuantize(
            dim=embedding_dim,
            codebook_size=codebook_size,
            decay=vq_decay,
            commitment_weight=commitment_weight,
            freeze_codebook=freeze_codebook,
        )
        self.vq.eval()
        self.vq.to(self.device)

        if vq_path is not None:
            state = torch.load(vq_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.vq.load_state_dict(state, strict=False)
        else:
            if freeze_codebook:
                print("Warning: VQ codebook is not provided. Tokens will be random.")

    def _lazy_torchaudio(self):
        import torchaudio

        return torchaudio

    def _to_mono(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 1:
            return wav.unsqueeze(0)
        if wav.shape[0] == 1:
            return wav
        return wav.mean(dim=0, keepdim=True)

    def _resample(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sample_rate:
            return wav
        ta = self._lazy_torchaudio()
        return ta.functional.resample(wav, sr, self.sample_rate)

    def _maybe_time_resample(self, feats: torch.Tensor, target_length: Optional[int]) -> torch.Tensor:
        if target_length is None:
            return feats
        if feats.shape[1] == target_length:
            return feats
        feats_t = feats.transpose(1, 2)
        feats_t = F.interpolate(feats_t, size=target_length, mode="nearest")
        return feats_t.transpose(1, 2)

    @torch.no_grad()
    def extract(
        self,
        wav: torch.Tensor,
        sample_rate: int,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        wav = self._to_mono(wav)
        wav = self._resample(wav, sample_rate)
        wav = wav.to(self.device)

        outputs = self.model(wav, output_hidden_states=True)
        if hasattr(outputs, "last_hidden_state"):
            feats = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            feats = outputs.hidden_states[-1]
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            feats = outputs[0]
        else:
            raise RuntimeError("MuQ outputs do not contain hidden states.")

        if feats.shape[-1] != self.embedding_dim:
            raise ValueError(f"Unexpected embedding dim: {feats.shape[-1]} (expected {self.embedding_dim})")

        feats = self._maybe_time_resample(feats, target_length)
        _, indices, _ = self.vq(feats)
        tokens = indices[0].to(torch.long).cpu()
        return tokens
