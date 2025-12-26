import argparse
import json
import os
from typing import Iterable, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from vector_quantize_pytorch import VectorQuantize


def _lazy_torchaudio():
    import torchaudio

    return torchaudio


def _cycle(iterable: Iterable):
    while True:
        for item in iterable:
            yield item


def _default_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    ta = _lazy_torchaudio()
    return ta.functional.resample(wav, sr, target_sr)


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        return wav.unsqueeze(0)
    if wav.shape[0] == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def _crop_wav(wav: torch.Tensor, sample_rate: int, duration_sec: Optional[float], strategy: str) -> torch.Tensor:
    if duration_sec is None:
        return wav
    target_len = int(duration_sec * sample_rate)
    if wav.shape[-1] <= target_len:
        return wav
    if strategy == "random":
        start = torch.randint(0, wav.shape[-1] - target_len + 1, (1,)).item()
    else:
        start = 0
    return wav[..., start : start + target_len]


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MuQAudioDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        sample_rate: int,
        max_duration: Optional[float],
        segment_strategy: str,
    ):
        self.items = load_jsonl(jsonl_path)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.segment_strategy = segment_strategy

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.items[idx]
        ta = _lazy_torchaudio()
        wav, sr = ta.load(item["audio_path"])
        wav = _resample(wav, sr, self.sample_rate)
        wav = _to_mono(wav)
        wav = _crop_wav(wav, self.sample_rate, self.max_duration, self.segment_strategy)
        return wav


def _load_muq(model_id: str, device: torch.device):
    try:
        from muq import MuQ
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("MuQ is not installed. Install it with `pip install muq`.") from exc
    model = MuQ.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return model


def _extract_embeddings(model, wav: torch.Tensor) -> torch.Tensor:
    outputs = model(wav, output_hidden_states=True)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        return outputs.hidden_states[-1]
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise RuntimeError("MuQ outputs do not contain hidden states.")


def train_vq(
    embedding_iter: Iterator[torch.Tensor],
    embedding_dim: int,
    codebook_size: int,
    steps: int,
    lr: float,
    decay: float,
    commitment_weight: float,
    device: torch.device,
) -> VectorQuantize:
    vq = VectorQuantize(
        dim=embedding_dim,
        codebook_size=codebook_size,
        decay=decay,
        commitment_weight=commitment_weight,
        freeze_codebook=False,
        learnable_codebook=False,
        ema_update=True,
    ).to(device)
    vq.train()

    params = list(vq.parameters())
    optimizer = torch.optim.Adam(params, lr=lr) if params else None

    for step in range(1, steps + 1):
        feats = next(embedding_iter).to(device)
        _, _, loss = vq(feats)
        loss = loss.mean()
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"[vq] step={step} loss={loss.item():.4f}")

    return vq


def save_vq(path: str, vq: VectorQuantize, config: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": vq.state_dict(), "config": config}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-jsonl", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--segment-strategy", type=str, default="random", choices=["random", "start"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--commitment-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = _default_device(args.device)
    model = _load_muq(args.model_id, device)

    dataset = MuQAudioDataset(
        jsonl_path=args.data_jsonl,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        segment_strategy=args.segment_strategy,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    embedding_iter = _cycle(dataloader)

    def embedding_generator():
        for wav in embedding_iter:
            wav = wav.to(device)
            with torch.no_grad():
                feats = _extract_embeddings(model, wav)
            yield feats.detach().cpu()

    vq = train_vq(
        embedding_iter=embedding_generator(),
        embedding_dim=args.embedding_dim,
        codebook_size=args.codebook_size,
        steps=args.steps,
        lr=args.lr,
        decay=args.decay,
        commitment_weight=args.commitment_weight,
        device=device,
    )

    config = {
        "model_id": args.model_id,
        "sample_rate": args.sample_rate,
        "embedding_dim": args.embedding_dim,
        "codebook_size": args.codebook_size,
        "decay": args.decay,
        "commitment_weight": args.commitment_weight,
        "steps": args.steps,
    }
    save_vq(args.output_path, vq, config)
    print(f"saved vq -> {args.output_path}")


if __name__ == "__main__":
    main()
