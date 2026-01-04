# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

SongBloomは、自己回帰スケッチと拡散ベースの精緻化を交互に行う新しいパラダイムを用いた、フルレングスの楽曲生成AIフレームワークです（2Bパラメータモデル）。10秒のオーディオプロンプトと歌詞から、最大4分の楽曲を生成できます。

**ライセンス**: 学術目的のみ使用可能。商用利用は禁止されています。

## 開発環境のセットアップ

### 初回セットアップ

```bash
# Python 3.10-3.12 が必要（3.13はPyTorch 2.2.0非対応）
uv python pin 3.12

# 依存関係のインストール
uv sync

# 環境変数の設定（推論実行前に必須）
source set_env.sh
```

### 依存関係の追加

```bash
uv add <package-name>
```

### Mac Silicon (Apple M1/M2/M3) 向け

```bash
# 環境変数を設定
export PYTORCH_ENABLE_MPS_FALLBACK=1
export DISABLE_FLASH_ATTN=1

# 推論実行（float32必須、bfloat16は非対応）
source set_env.sh
uv run python infer.py --input-jsonl example/test.jsonl --device mps --dtype float32
```

### トラブルシューティング

#### Python 3.13 エラー
PyTorch 2.2.0 は Python 3.13 に対応していません：
```bash
uv python pin 3.12
uv sync
```

#### NumPy 2.x 互換性エラー
`numpy.dtype size changed` エラーが出る場合、NumPy 1.x が必要です。
pyproject.toml には既に `numpy<2` が設定済み。

## 推論コマンド

```bash
# 基本的な推論
uv run python infer.py --input-jsonl example/test.jsonl

# 低VRAMのGPU向け（RTX4090など）
uv run python infer.py --input-jsonl example/test.jsonl --dtype bfloat16

# 利用可能なモデル: songbloom_full_150s, songbloom_full_150s_dpo, songbloom_full_240s
uv run python infer.py --model-name songbloom_full_240s --input-jsonl example/test_240s.jsonl
```

## アーキテクチャ

### コアコンポーネント

- **`infer.py`**: メイン推論スクリプト。Hugging Faceからモデルをダウンロードし、JSONL入力を処理
- **`SongBloom/models/songbloom/songbloom_pl.py`**: `SongBloom_Sampler`クラス（推論ラッパー）と`SongBloom_PL`（PyTorch Lightningモジュール）
- **`SongBloom/models/songbloom/songbloom_mvsa.py`**: `MVSA_DiTAR`モデル（マルチストリームVAE + 自己回帰トランスフォーマー）
- **`SongBloom/models/transformer.py`**: コアDiT（Diffusion Transformer）実装
- **`SongBloom/models/vae_frontend/autoencoders.py`**: `StableVAE`オーディオ圧縮モデル
- **`SongBloom/g2p/`**: Grapheme-to-Phoneme変換（中国語、英語、日本語をサポート）

### データフォーマット

入力JSONLの必須フィールド:
```json
{"idx": "sample_id", "lyrics": "歌詞テキスト", "prompt_wav": "10秒48kHzのWAVパス"}
```

### 歌詞フォーマット規則

- ボーカルセクション: `[verse]`、`[chorus]`、`[bridge]` のプレフィックスが必要
- 非ボーカルセクション: `[intro]`、`[inst]`、`[outro]` を長さに応じて繰り返す
  - 150sモデル: 各タグ = 1秒
  - 240sモデル: 各タグ = 5秒
- 文の区切りはピリオド(`.`)、セクションの区切りはカンマ(`,`)
- 詳細: `docs/lyric_format.md`

例:
```
[intro] [intro] [intro] , [verse] 歌詞テキスト. 次の文. , [chorus] サビの歌詞. , [outro] [outro]
```

## ディレクトリ構造

- `cache/`: モデルのダウンロード先（自動生成）
- `output/`: 生成されたFLACファイルの出力先
- `example/`: サンプル入力ファイル（test.jsonl、test.wav）
- `SongBloom/`: メインパッケージ

## 学習（Fine-tuning）

### 重要な注意事項

> **公式学習コードは未公開です**

- Tencent AI Lab は企業ポリシーにより学習コードを公開していません
- このリポジトリの学習コードは**論文に基づく独自実装**です
- 参照: [GitHub Issue #29](https://github.com/tencent-ailab/SongBloom/issues/29)

### 学習コマンド

```bash
# 日本語Fine-tuning（Apple Silicon向け）
uv run python train_japanese.py \
    --data-jsonl data/japanese_songs.jsonl \
    --val-split 0.05 \
    --device mps \
    --precision 32 \
    --batch-size 1 \
    --accumulate-grad-batches 8 \
    --init-from-pretrained \
    --use-cache \
    --use-lora

# CUDA GPU向け
uv run python train_japanese.py \
    --data-jsonl data/japanese_songs.jsonl \
    --val-split 0.05 \
    --device cuda \
    --precision 16-mixed \
    --init-from-pretrained \
    --use-cache \
    --use-lora
```

### 学習データフォーマット

学習には以下の形式のJSONLファイルが必要：

```json
{
  "idx": "song_001",
  "audio_path": "/path/to/song.wav",
  "lyrics": "[intro] [intro] , [verse] 歌詞テキスト. , [chorus] サビ. , [outro]",
  "prompt_wav": "/path/to/10sec_prompt.wav",
  "sketch_path": "/path/to/sketch_tokens.pt"
}
```

- `audio_path`: 学習対象の楽曲（48kHz、ステレオ、最大150秒）
- `lyrics`: 構造タグ付き歌詞
- `prompt_wav`: 10秒のリファレンス音声
- `sketch_path`: スケッチトークン（事前計算、任意）

詳細は `docs/training_data_format.md` を参照。

### 学習アーキテクチャ

- **スケッチトークン抽出**: MuQ (SSL音楽モデル) + Vector Quantization
- **アコースティック潜在変数**: StableVAE エンコーダー
- **損失関数**: `L_total = L_LM + 0.1 × L_flow`
  - `L_LM`: CrossEntropy（ARステージ、スケッチトークン予測）
  - `L_flow`: MSE（NARステージ、Rectified Flow）
- **メモリ効率化**: LoRA（rank=16, alpha=32）

詳細は `docs/training_architecture.md` を参照。

### 学習用追加依存

```bash
uv add muq  # MuQスケッチ抽出
```
