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
- **`SongBloom/g2p/`**: Grapheme-to-Phoneme変換（中国語と英語をサポート）

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
