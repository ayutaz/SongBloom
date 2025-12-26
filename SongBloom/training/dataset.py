import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from .sketch import PrecomputedSketchExtractor, SketchExtractor
from ..g2p.lyric_common import key2processor
from normalize_lyrics import clean_lyrics

if TYPE_CHECKING:
    from ..models.vae_frontend import StableVAE


@dataclass
class DatasetConfig:
    jsonl_path: str
    sample_rate: int = 48000
    prompt_len: float = 10.0
    max_duration: Optional[float] = None
    cache_dir: Optional[str] = None
    use_cache: bool = True
    rebuild_cache: bool = False
    segment_strategy: str = "start"  # start | random
    clean_lyrics: bool = False
    process_lyrics: bool = False
    lyric_processor: Optional[str] = None


def _load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _lazy_torchaudio():
    import torchaudio

    return torchaudio


def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    ta = _lazy_torchaudio()
    return ta.functional.resample(wav, sr, target_sr)


def _crop_wav(wav: torch.Tensor, sample_rate: int, max_duration: Optional[float], strategy: str) -> torch.Tensor:
    if max_duration is None:
        return wav
    max_len = int(max_duration * sample_rate)
    if wav.shape[-1] <= max_len:
        return wav
    if strategy == "random":
        start = torch.randint(0, wav.shape[-1] - max_len + 1, (1,)).item()
    else:
        start = 0
    return wav[..., start:start + max_len]


def _pad_or_trim_prompt(wav: torch.Tensor, sample_rate: int, prompt_len: float) -> torch.Tensor:
    target_len = int(prompt_len * sample_rate)
    if wav.shape[-1] >= target_len:
        return wav[..., :target_len]
    pad_len = target_len - wav.shape[-1]
    pad = torch.zeros(wav.shape[0], pad_len, device=wav.device, dtype=wav.dtype)
    return torch.cat([wav, pad], dim=-1)


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        return wav.unsqueeze(0)
    if wav.shape[0] == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def _trim_to_block_multiple(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int]:
    length = x.shape[-1]
    new_len = (length // block_size) * block_size
    if new_len <= 0:
        return x[..., :0], 0
    return x[..., :new_len], new_len


class SongBloomTrainDataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfig,
        vae: "StableVAE",
        block_size: int,
        sketch_extractor: Optional[SketchExtractor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.items = _load_jsonl(cfg.jsonl_path)
        self.vae = vae
        self.block_size = block_size
        self.sketch_extractor = sketch_extractor
        if cfg.cache_dir:
            os.makedirs(cfg.cache_dir, exist_ok=True)
        self._lyric_processor = key2processor.get(cfg.lyric_processor) if cfg.lyric_processor else None

    def __len__(self) -> int:
        return len(self.items)

    def _cache_path(self, idx: str) -> Optional[str]:
        if not self.cfg.cache_dir:
            return None
        return os.path.join(self.cfg.cache_dir, f"{idx}.pt")

    def _load_audio(self, path: str) -> torch.Tensor:
        ta = _lazy_torchaudio()
        wav, sr = ta.load(path)
        wav = _resample_if_needed(wav, sr, self.cfg.sample_rate)
        wav = _crop_wav(wav, self.cfg.sample_rate, self.cfg.max_duration, self.cfg.segment_strategy)
        return wav

    def _load_prompt(self, path: str) -> torch.Tensor:
        ta = _lazy_torchaudio()
        wav, sr = ta.load(path)
        wav = _resample_if_needed(wav, sr, self.cfg.sample_rate)
        wav = _to_mono(wav)
        wav = _pad_or_trim_prompt(wav, self.cfg.sample_rate, self.cfg.prompt_len)
        return wav

    def _process_lyrics(self, lyrics: str) -> str:
        if self.cfg.clean_lyrics:
            lyrics = clean_lyrics(lyrics)
        if not self.cfg.process_lyrics or self._lyric_processor is None:
            return lyrics
        return self._lyric_processor(lyrics)

    def _load_sketch_tokens(self, item: dict, wav: torch.Tensor, target_length: Optional[int]) -> torch.Tensor:
        if "sketch_tokens" in item:
            tokens = torch.as_tensor(item["sketch_tokens"], dtype=torch.long)
            return tokens.reshape(-1)
        if self.sketch_extractor is None:
            extractor = PrecomputedSketchExtractor()
            return extractor.extract_from_item(item)
        # external extractor path
        return self.sketch_extractor.extract(wav, self.cfg.sample_rate, target_length=target_length)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        idx = item.get("idx", str(index))
        cache_path = self._cache_path(idx)
        if cache_path and self.cfg.use_cache and os.path.exists(cache_path) and not self.cfg.rebuild_cache:
            cached = torch.load(cache_path, map_location="cpu")
            return {
                "idx": idx,
                "lyrics": cached["lyrics"],
                "prompt_wav": self._load_prompt(item["prompt_wav"]),
                "audio_latent": cached["audio_latent"],
                "sketch_tokens": cached["sketch_tokens"],
                "length": cached["length"],
            }

        lyrics = self._process_lyrics(item["lyrics"])
        audio = self._load_audio(item["audio_path"])
        prompt = self._load_prompt(item["prompt_wav"])

        with torch.no_grad():
            audio_latent = self.vae.encode(audio.unsqueeze(0)).squeeze(0)  # [D, T]

        sketch_tokens = self._load_sketch_tokens(item, audio, target_length=audio_latent.shape[-1])

        # align lengths
        min_len = min(audio_latent.shape[-1], sketch_tokens.shape[-1])
        audio_latent = audio_latent[..., :min_len]
        sketch_tokens = sketch_tokens[:min_len]

        audio_latent, x_len = _trim_to_block_multiple(audio_latent, self.block_size)
        sketch_tokens = sketch_tokens[:x_len]

        out = {
            "idx": idx,
            "lyrics": lyrics,
            "prompt_wav": prompt,
            "audio_latent": audio_latent,
            "sketch_tokens": sketch_tokens,
            "length": x_len,
        }
        if cache_path and self.cfg.use_cache:
            torch.save(
                {
                    "lyrics": lyrics,
                    "audio_latent": audio_latent,
                    "sketch_tokens": sketch_tokens,
                    "length": x_len,
                },
                cache_path,
            )
        return out


def collate_training_batch(batch: List[dict], sketch_pad_value: int = 0) -> dict:
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item()) if lengths.numel() else 0

    # audio_latent: [D, T]
    latent_dim = batch[0]["audio_latent"].shape[0] if max_len > 0 else 0
    latents = torch.zeros(len(batch), latent_dim, max_len, dtype=batch[0]["audio_latent"].dtype)
    sketches = torch.full((len(batch), max_len), sketch_pad_value, dtype=torch.long)

    for i, b in enumerate(batch):
        length = b["length"]
        if length == 0:
            continue
        latents[i, :, :length] = b["audio_latent"][:, :length]
        sketches[i, :length] = b["sketch_tokens"][:length]

    prompts = torch.stack([b["prompt_wav"] for b in batch], dim=0)  # [B, 1, T]
    lyrics = [b["lyrics"] for b in batch]
    idx = [b["idx"] for b in batch]

    return {
        "idx": idx,
        "lyrics": lyrics,
        "prompt_wav": prompts,
        "audio_latent": latents,
        "sketch_tokens": sketches,
        "lengths": lengths,
    }
