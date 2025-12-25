# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

SongBloomは、自己回帰スケッチと拡散ベースの精緻化を交互に行う新しいパラダイムを用いた、フルレングスの楽曲生成AIフレームワークです（2Bパラメータモデル）。10秒のオーディオプロンプトと歌詞から、最大4分の楽曲を生成できます。

**ライセンス**: 学術目的のみ使用可能。商用利用は禁止されています。

## 開発環境のセットアップ

```bash
# uvで依存関係をインストール（Python 3.10-3.12が必要）
uv sync

# 環境変数の設定（実行前に必須）
source set_env.sh
```

### 依存関係の追加

```bash
uv add <package-name>
```

## 推論コマンド

```bash
# 基本的な推論
uv run python infer.py --input-jsonl example/test.jsonl

# 低VRAMのGPU向け（RTX4090など）
uv run python infer.py --input-jsonl example/test.jsonl --dtype bfloat16

# 利用可能なモデル: songbloom_full_150s, songbloom_full_150s_dpo, songbloom_full_240s
uv run python infer.py --model-name songbloom_full_240s --input-jsonl example/test_240s.jsonl
```

### Mac Silicon向け

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export DISABLE_FLASH_ATTN=1
# dtype は float32 を使用（bfloat16は非対応）
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
