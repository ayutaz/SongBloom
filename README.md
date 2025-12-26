

<p align="center"><img src="docs/icon.png" width="50%"></p>


# **SongBloom**: *Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement*

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2506.07634-b31b1b.svg)](https://arxiv.org/abs/2506.07634)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/CypressYang/SongBloom)
[![Demo Page](https://img.shields.io/badge/Demo-Audio%20Samples-green)](https://cypress-yang.github.io/SongBloom_demo)

</div>

We propose **SongBloom**, a novel framework for full-length song generation that leverages an interleaved paradigm of autoregressive sketching and diffusion-based refinement. SongBloom employs an autoregressive diffusion model that combines the high fidelity of diffusion models with the scalability of language models.
Specifically, it gradually extends a musical sketch from short to long and refines the details from coarse to fine-grained. The interleaved generation paradigm effectively integrates prior semantic and acoustic context to guide the generation process.
Experimental results demonstrate that SongBloom outperforms existing methods across both subjective and objective metrics and achieves performance comparable to the state-of-the-art commercial music generation platforms.

![img](docs/architecture.png)



## Models

| Name                 | Size | Max Length | Prompt type | 🤗                                            |
| -------------------- | ---- | ---------- | ----------- | -------------------------------------------- |
| songbloom_full_150s  | 2B   | 2m30s      | 10s wav     | [link](https://huggingface.co/CypressYang/SongBloom) |
| songbloom_full_150s_dpo  | 2B   | 2m30s      | 10s wav     | [link](https://huggingface.co/CypressYang/SongBloom) |
| songbloom_full_240s$^{[1]}$ | 2B   | 4m     | 10s wav |        [link](https://huggingface.co/CypressYang/SongBloom_long)                           |
| ... |      |            |             |                                              |

- [1] For the **_150s** series models, each `[intro]`, `[outro]`, and `[inst]` corresponds to an expected duration of 1 second; whereas for the **_240s** series models, each token corresponds to 5 seconds (details in [docs/lyric_format](docs/lyric_format.md)).

## Updates
- **Oct 2025**: Release songbloom_full_240s; fix bugs in half-precision inference ; Reduce GPU memory consumption during the VAE stage.
- **Sep 2025**: Release the songbloom_full_150s model with DPO post-training
- **Jun 2025**: Release the songbloom_full_150s and inference script




## Getting Started

### Prepare Environments

```bash
# Install dependencies (requires Python 3.10-3.12)
uv sync

# For CUDA support, modify pyproject.toml index URL:
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cu118"
```

### Data Preparation

A  .jsonl file, where each line is a json object:

```json
{
	"idx": "The index of each sample", 
	"lyrics": "The lyrics to be generated",
	"prompt_wav": "The path of the style prompt audio",
}
```

One example can be refered to as: [example/test.jsonl](example/test.jsonl)

The prompt wav should be a 10-second, 48kHz audio clip.

For details on lyric formatting, see [docs/lyric_format.md](docs/lyric_format.md).

### Inference

```bash
source set_env.sh

uv run python infer.py --input-jsonl example/test.jsonl

# For GPUs with low VRAM like RTX4090, you should set the dtype as bfloat16
uv run python infer.py --input-jsonl example/test.jsonl --dtype bfloat16

# Output WAV or MP3 (MP3 requires FFmpeg backend in torchaudio)
uv run python infer.py --input-jsonl example/test.jsonl --output-format wav

# Use a fine-tuned Lightning checkpoint (LoRA optional)
uv run python infer.py --input-jsonl example/test.jsonl \
  --ckpt-path checkpoints/jacappella_muq_lora/last.ckpt \
  --use-lora

# SongBloom also supports flash-attn (optional). To enable it, please install flash-attn (v2.6.3 is used during training) manually and set os.environ['DISABLE_FLASH_ATTN'] = "0" in infer.py:8
```

- model-name: Specify model version, see the model cards (eg: songbloom_full_150s/songbloom_full_150s_dpo);
- local-dir: Dir where the weights and config files are downloaded;
- input-jsonl: input raw data;
- output-dir: Dir where the output audio saved;
- n-samples: How many audios will be generated for each input term;
- output-format: Audio output format (flac/wav/mp3; mp3 requires FFmpeg backend);
- ckpt-path: Use a fine-tuned Lightning checkpoint instead of the base model;
- use-lora: Enable LoRA modules when loading a fine-tuned checkpoint;

## Mac Silicon

Set these environment variables before running:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export DISABLE_FLASH_ATTN=1
```

Run inference with MPS device and float32 (bfloat16 is not supported on MPS):

```bash
source set_env.sh
uv run python infer.py --input-jsonl example/test.jsonl --device mps --dtype float32
```

When loading the model programmatically, explicitly pass the MPS device:

```python
import torch

device = torch.device('mps')
model = SongBloom_Sampler.build_from_trainer(cfg, strict=False, dtype=torch.float32, device=device)
```

## Troubleshooting

### Python 3.13 Not Supported
PyTorch 2.2.0 requires Python 3.10-3.12. If using Python 3.13:
```bash
uv python pin 3.12
uv sync
```

### NumPy 2.x Compatibility Error
If you see `numpy.dtype size changed` error, NumPy 1.x is required.
This is already configured in pyproject.toml with `numpy<2`.

## Code Quality

Ruff is configured via `pyproject.toml`.
To lint:

```bash
ruff check .
```

## Training (Fine-tuning)

This repository now includes a minimal training pipeline (Lightning) aimed at Japanese fine-tuning.
See `docs/training_code.md` for details, JSONL format, and required sketch tokens.
W&B logging is enabled by default in `train_japanese.py` (disable with `--no-wandb`).
jaCappella can be used to bootstrap a lyrics-aligned Japanese dataset (see `docs/training_code.md`).

## Citation

```
@article{yang2025songbloom,
title={SongBloom: Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement},
author={Yang, Chenyu and Wang, Shuai and Chen, Hangting and Tan, Wei and Yu, Jianwei and Li, Haizhou},
journal={arXiv preprint arXiv:2506.07634},
year={2025}
}
```

## License

SongBloom (codes and weights) is released under the [LICENSE](LICENSE). 
