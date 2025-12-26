# 学習コード（日本語ファインチューニング）

このリポジトリには推論コードのみが含まれていたため、学習用の最小実装を追加しました。
本ドキュメントは **日本語楽曲の生成を目的としたファインチューニング** を想定した使い方の説明です。

## 重要な前提（スケッチトークン）

SongBloomの学習には **MuQ + VQ によるスケッチトークン** が必要ですが、このリポジトリには MuQ の実装が含まれていません。
そのため、学習コードは以下のどちらかを要求します：

- **事前計算したスケッチトークンをJSONLで渡す**（推奨）
- `SongBloom/training/sketch.py` を自分のMuQパイプラインで実装する

現在の実装は `sketch_path` で読み込む方式を標準としています。
MuQを使って自動抽出する場合は `--sketch-mode muq` を指定してください（下記）。

## JSONLフォーマット（学習）

既存の `docs/training_data_format.md` に加えて、**sketch tokens の参照**を追加してください。

例:

```json
{"idx":"song_001","audio_path":"data/audio/song_001.wav","lyrics":"[intro] [intro] , [verse] 歌詞...","prompt_wav":"data/prompts/song_001_prompt.wav","sketch_path":"data/sketch/song_001.pt"}
```

- `sketch_path` は `.pt` / `.npy` / `.npz` に対応
- `sketch_path` の中身は **1次元の整数トークン列**（shape [T]）
- `.pt` の場合、`{"sketch": tensor}` 形式もOK

## 学習実行

```bash
source set_env.sh

# 例: 150秒モデルを日本語データで微調整
uv run python train_japanese.py \
  --model-name songbloom_full_150s \
  --data-jsonl data/japanese_songs.jsonl \
  --val-split 0.05 \
  --output-dir checkpoints \
  --batch-size 1 \
  --accumulate-grad-batches 8 \
  --device mps \
  --precision 32 \
  --init-from-pretrained \
  --use-cache
```

### 主なオプション

- `--use-cache` : `cache/training/` に VAE latent と sketch を保存
- `--rebuild-cache` : キャッシュ再生成
- `--segment-strategy start|random` : 長尺音声の切り出し方法
- `--process-lyrics` : データセット側で G2P 変換
  - デフォルトは **学習モジュール側で変換**（推奨）
- `--val-jsonl` / `--val-split` : 検証データの指定
- `--resume-from` : チェックポイントから再開
- `--require-vq-path` : MuQ使用時に VQ 重みの指定を必須化
- `--verify-lengths` : MuQ出力とVAE latent長のズレを検証（先頭N件）
- `--verify-lengths-max` : 検証する最大件数
- `--require-length-match` : 長さ不一致があればエラー
- `--no-wandb` : W&Bロギングを無効化
- `--wandb-project` / `--wandb-entity` / `--wandb-name` / `--wandb-mode` : W&B設定

### JSONL分割ユーティリティ

```bash
uv run python -m SongBloom.training.split_jsonl \
  --input-jsonl data/japanese_songs.jsonl \
  --train-jsonl data/japanese_songs_train.jsonl \
  --val-jsonl data/japanese_songs_val.jsonl \
  --val-ratio 0.05
```

## 追加したコード

- `SongBloom/training/dataset.py` : 学習データセット + キャッシュ
- `SongBloom/training/sketch.py` : スケッチ抽出（現在は事前計算前提）
- `train_japanese.py` : 学習スクリプト（Lightning）

## MuQを使ってスケッチを計算したい場合

`--sketch-mode muq` を指定すると、MuQ + VQ でスケッチトークンを生成します。
MuQは別途インストールが必要です。

```bash
pip install muq
```

必要に応じて `--muq-vq-path` に VQ の事前学習済み重みを指定してください。
再現性重視の場合は `--require-vq-path` を付けて必須化してください。
MuQの出力は 25fps を想定し、VQ後のトークン長が VAE latent と合うように調整してください。

---

この学習コードは「論文の手順に沿った最小構成」です。
本格的に再現する場合は `docs/training_architecture.md` の前処理パイプラインも併せて実装してください。
