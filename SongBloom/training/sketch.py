import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SketchSource:
    """Configuration for loading or computing sketch tokens."""
    mode: str = "precomputed"  # precomputed | external
    field: str = "sketch_path"


class SketchExtractor:
    """Abstract sketch extractor.

    Implementations should return a LongTensor of shape [T].
    """

    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:  # pragma: no cover - interface
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


class ExternalSketchExtractor(SketchExtractor):
    """Placeholder for external sketch extraction (e.g., MuQ + VQ).

    Replace this with your MuQ pipeline, or precompute tokens and use
    PrecomputedSketchExtractor instead.
    """

    def __init__(self):
        raise NotImplementedError(
            "External sketch extraction is not wired in this repo. "
            "Provide precomputed sketch tokens via 'sketch_path' in JSONL, "
            "or implement your MuQ+VQ pipeline here."
        )
