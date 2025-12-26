import argparse
import json
import os
import wave
from typing import List

import numpy as np


def _write_wav(path: str, sample_rate: int, duration_sec: float, rng: np.random.Generator) -> None:
    num_frames = int(duration_sec * sample_rate)
    audio = rng.uniform(-0.2, 0.2, size=(num_frames,)).astype(np.float32)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _save_sketch(path: str, tokens: np.ndarray) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        import torch

        torch.save(torch.from_numpy(tokens).long(), path)
    elif ext in (".npy", ".npz"):
        np.save(path, tokens)
    else:
        raise ValueError(f"Unsupported sketch format: {path}")


def _compute_sketch_len(duration_sec: float, fps: int) -> int:
    return max(1, int(round(duration_sec * fps)))


def make_dummy_dataset(
    output_dir: str,
    num_samples: int,
    duration_sec: float,
    sample_rate: int,
    prompt_sec: float,
    sketch_fps: int,
    codebook_size: int,
    sketch_ext: str,
    seed: int,
) -> List[dict]:
    rng = np.random.default_rng(seed)
    audio_dir = os.path.join(output_dir, "audio")
    prompt_dir = os.path.join(output_dir, "prompts")
    sketch_dir = os.path.join(output_dir, "sketch")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(sketch_dir, exist_ok=True)

    items = []
    for i in range(num_samples):
        idx = f"dummy_{i:04d}"
        audio_path = os.path.join(audio_dir, f"{idx}.wav")
        prompt_path = os.path.join(prompt_dir, f"{idx}_prompt.wav")
        sketch_path = os.path.join(sketch_dir, f"{idx}{sketch_ext}")

        _write_wav(audio_path, sample_rate, duration_sec, rng)
        _write_wav(prompt_path, sample_rate, prompt_sec, rng)

        sketch_len = _compute_sketch_len(duration_sec, sketch_fps)
        sketch_tokens = rng.integers(0, codebook_size, size=(sketch_len,), dtype=np.int64)
        _save_sketch(sketch_path, sketch_tokens)

        items.append(
            {
                "idx": idx,
                "audio_path": audio_path,
                "lyrics": "[verse] ダミー. , [chorus] ダミー.",
                "prompt_wav": prompt_path,
                "sketch_path": sketch_path,
            }
        )

    return items


def write_jsonl(path: str, items: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--prompt-sec", type=float, default=10.0)
    parser.add_argument("--sketch-fps", type=int, default=25)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--sketch-ext", type=str, default=".npy", choices=[".npy", ".npz", ".pt"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--jsonl-path", type=str, default=None)
    args = parser.parse_args()

    items = make_dummy_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        duration_sec=args.duration_sec,
        sample_rate=args.sample_rate,
        prompt_sec=args.prompt_sec,
        sketch_fps=args.sketch_fps,
        codebook_size=args.codebook_size,
        sketch_ext=args.sketch_ext,
        seed=args.seed,
    )

    jsonl_path = args.jsonl_path or os.path.join(args.output_dir, "dummy.jsonl")
    write_jsonl(jsonl_path, items)
    print(f"wrote {len(items)} samples -> {jsonl_path}")


if __name__ == "__main__":
    main()
