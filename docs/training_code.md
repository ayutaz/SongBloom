# 学習コード（日本語ファインチューニング）

このリポジトリには推論コードのみが含まれていたため、学習用の最小実装を追加しました。

最新の進捗と課題は以下にまとめています:
- `docs/status_report_ja.md`
本ドキュメントは **日本語楽曲の生成を目的としたファインチューニング** を想定した使い方の説明です。

## 重要な前提（スケッチトークン）

SongBloomの学習には **MuQ + VQ によるスケッチトークン** が必要です。
このリポジトリでは以下のどちらかを選べます：

- **事前計算したスケッチトークンをJSONLで渡す**（推奨）
- `--sketch-mode muq` で学習時に自動抽出する（MuQの別途インストールが必要）

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

### 監視（W&B）

学習ログは **W&B がデフォルトで有効** です。無効化する場合は `--no-wandb` を指定してください。

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
- `--use-lora` : LoRAを有効化（メモリ削減）
- `--lora-rank` / `--lora-alpha` / `--lora-dropout` : LoRAハイパーパラメータ
- `--lora-target-modules` : LoRAを当てるモジュール名（カンマ区切り）
- `--lora-train-all` : LoRA以外も学習（通常はLoRAのみ学習）

### JSONL分割ユーティリティ

```bash
uv run python -m SongBloom.training.split_jsonl \
  --input-jsonl data/japanese_songs.jsonl \
  --train-jsonl data/japanese_songs_train.jsonl \
  --val-jsonl data/japanese_songs_val.jsonl \
  --val-ratio 0.05
```

### ダミーデータ生成（実データがない場合の検証用）

```bash
uv run python -m SongBloom.training.make_dummy_dataset \
  --output-dir data/dummy \
  --num-samples 4 \
  --duration-sec 30 \
  --prompt-sec 10 \
  --sketch-fps 25
```

生成された `data/dummy/dummy.jsonl` を `--data-jsonl` に指定して、\n
`--val-split` や `--verify-lengths` の動作確認が可能です。

### jaCappella からデータセットを作成

jaCappella は **歌詞(MusicXML)付き**のため、手作業なしで JSONL を生成できます。
Hugging Face 側で利用条件の同意が必要です（ログイン後に承諾）。

```bash
uv run python -m SongBloom.training.prepare_jacappella \
  --output-dir data/jacappella_prepared \
  --download-dir data/jacappella_raw \
  --audio-type lead_vocal \
  --musicxml-type svs \
  --prompt-sec 10 \
  --clean-japanese
```

生成された `data/jacappella_prepared/jacappella.jsonl` を `--data-jsonl` に指定してください。
※ `--clean-japanese` は英字/不要記号を除去し、日本語のみの歌詞に整形します。

### MuQ + VQ のコードブック学習

SongBloomの精度を上げるには、MuQ埋め込み用の VQコードブック（16384）が必要です。
以下で VQ 重みを学習し、`--muq-vq-path` に指定してください。

```bash
uv run python -m SongBloom.training.train_vq \
  --data-jsonl data/jacappella_prepared/jacappella.jsonl \
  --output-path data/vq/vq_16384.pt \
  --steps 200 \
  --batch-size 2 \
  --device cpu
```

※ MuQ の前処理で複素数演算が発生するため、MPS ではエラーになることがあります。
その場合は `--device cpu` を指定してください。

## 追加したコード

- `SongBloom/training/dataset.py` : 学習データセット + キャッシュ
- `SongBloom/training/sketch.py` : スケッチ抽出（MuQ + VQ）
- `SongBloom/training/split_jsonl.py` : JSONL分割ユーティリティ
- `train_japanese.py` : 学習スクリプト（Lightning / W&B / resume / val対応）

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
