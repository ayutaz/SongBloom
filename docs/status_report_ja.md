# 現状まとめ（2025-01-17 更新）

## 0. 重要な注意事項

> **公式学習コードは未公開です**

- Tencent AI Lab は企業ポリシーにより学習コードを公開していません
- 参照: [GitHub Issue #29](https://github.com/tencent-ailab/SongBloom/issues/29)
- このリポジトリの学習コードは**論文に基づく独自実装**です
- 動作は部分的に検証済みですが、公式サポートはありません

---

## 1. 目的
- 日本語歌詞での楽曲生成を可能にするため、MuQ+VQ スケッチと LoRA で SongBloom を微調整。
- jaCappella を用いた学習パイプラインの整備と推論検証。

## 2. 作業環境
- 端末: Apple M4 Max / 64GB RAM
- デバイス: MPS 使用（PyTorch）
- Python: 3.12.12
- PyTorch: 2.2.0
- 注意: MuQ の前処理に複素数演算が含まれるため、MPS では VQ 学習が失敗しやすい（CPU 推奨）。

## 3. 実装済みの主な機能
- MuQ + VQ スケッチ抽出:
  - `SongBloom/training/sketch.py` に ExternalSketchExtractor 実装。
  - 24kHz resample → MuQ 埋め込み(1024) → VQ(codebook=**129**)。
  - VAE latent と 25fps で長さ合わせ。
  - **重要**: コードブックサイズは129（モデル埋め込み層が130クラス対応のため）。
- データ分割:
  - JSONL を train/val に分割するユーティリティ追加。
  - Lightning に val dataloader を渡す対応。
- 学習再開:
  - `train_japanese.py` に `--resume-from` 追加。
  - Optimizer/Scheduler の復元に対応。
- LoRA 対応:
  - `SongBloom/training/lora.py` を追加。
  - `train_japanese.py` に LoRA オプションを追加。
- VQ 学習スクリプト:
  - `SongBloom/training/train_vq.py` 追加。
- 推論改善:
  - `infer.py` に `--output-format`（wav/mp3/flac）追加。
  - `infer.py` に `--ckpt-path` / `--use-lora` 追加。
- CI / 品質:
  - Ruff の導入と CI（GitHub Actions）での実行。
  - テスト追加（dataset, split, sketch, VQ, LoRA, prepare_jacappella）。

## 4. jaCappella のデータ準備
### 4.1 旧データ（ノイズ混入）
- `data/jacappella_prepared/jacappella.jsonl`
- 英字・擬音（LABADADA / TUTU など）が大量に混入していたため、推論時の音素が英語寄りに変換されていた。

### 4.2 日本語クリーニング対応（現状）
- `prepare_jacappella.py` に `--clean-japanese` を追加。
  - ひらがな/カタカナ/漢字/々/ー/・/句読点以外を除去。
- 生成済み JSONL:
  - `data/jacappella_prepared_ja/jacappella.jsonl`
  - 無効/空歌詞は 24 件スキップ済み。
  - 有効行数: **26**

コマンド例:
```bash
uv run python -m SongBloom.training.prepare_jacappella \
  --output-dir data/jacappella_prepared_ja \
  --download-dir data/jacappella_raw \
  --audio-type lead_vocal \
  --musicxml-type svs \
  --prompt-sec 10 \
  --clean-japanese \
  --no-download
```

## 5. VQ コードブック学習

> **重要**: コードブックサイズは **129** を使用してください。
> モデルの埋め込み層 (`skeleton_emb`) は130クラスのみサポートしています。
> 16384クラスを使用すると、損失が ln(16384) ≈ 9.7 で停滞します。

コマンド例:
```bash
uv run python train_vq_codebook.py \
  --data-dir data/japanese_singing_prepared/audio \
  --output checkpoints/vq_codebook.pt \
  --codebook-size 129 \
  --epochs 10 \
  --device cuda
```

**注意:**
- EMAベースのVQは5-10エポックで収束します（50エポックは不要）
- CPU / CUDA 推奨（MPS は複素数演算でエラー）

## 6. 学習（微調整）
### 6.1 旧データでの学習
- 出力先: `checkpoints/jacappella_muq_lora/`
- 目的: baseline 動作確認

### 6.2 japanese-singing-voice での学習（現状）
- 出力先: `checkpoints/japanese_singing_v4/`
- データセット: japanese-singing-voice (403曲, ~20時間)
- 主要設定:
  - batch=1 / accumulate=8 / max_epochs=100
  - `--sketch-mode muq` + `--muq-vq-path checkpoints/vq_codebook_129.pt`
  - `--muq-codebook-size 129`
  - `--use-lora`（rank=16, alpha=32）
  - `--strategy ddp_find_unused_parameters_true`（DDP + LoRA必須）

コマンド例（複数GPU）:
```bash
uv run python train_japanese.py \
  --data-jsonl data/japanese_singing_prepared/japanese_singing.jsonl \
  --val-split 0.1 \
  --output-dir checkpoints/japanese_singing_v4 \
  --batch-size 1 \
  --accumulate-grad-batches 8 \
  --max-epochs 100 \
  --device cuda \
  --devices 4 \
  --strategy ddp_find_unused_parameters_true \
  --precision 16-mixed \
  --sketch-mode muq \
  --muq-vq-path checkpoints/vq_codebook_129.pt \
  --muq-codebook-size 129 \
  --use-lora \
  --init-from-pretrained
```

## 7. 推論（検証）
### 7.1 フル長推論（150秒相当）
- `val_split` が2件、`n-samples=2` の場合は **30〜60分**かかる。

### 7.2 30秒・1本・1件（短縮検証）
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 DISABLE_FLASH_ATTN=1 \
./.venv/bin/python infer.py \
  --input-jsonl output/jacappella_finetune_ja/one_sample.jsonl \
  --ckpt-path checkpoints/jacappella_muq_lora_ja/last.ckpt \
  --use-lora \
  --device mps \
  --dtype float32 \
  --output-format wav \
  --output-dir output/jacappella_finetune_ja_30s \
  --n-samples 1 \
  --max-duration 30
```

生成物:
- `output/jacappella_finetune_ja_30s/*.wav`

## 8. 解決済みの問題

### 8.1 VQコードブックサイズの不整合（2025-01-09 解決）

**症状:**
- 損失が ln(16384) ≈ 9.7 で停滞し、学習が全く進まない

**根本原因:**
- モデルの埋め込み層 (`skeleton_emb`) は `num_pitch=128` + 特殊トークンで **130クラス** のみサポート
- VQコードブックが16384クラスだと、スケッチトークンが [0, 16383] の範囲になる
- 埋め込み層の範囲外のためトークン予測が不可能

**解決方法:**
- VQコードブックを **129クラス** で再学習
- `--muq-codebook-size 129` を指定

**結果:**
- 損失が 9.7 → **6.3-7.8** に改善
- 学習が正常に進行

### 8.2 DDP + LoRA の未使用パラメータエラー（2025-01-09 解決）

**症状:**
```
RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss
```

**解決方法:**
- `--strategy ddp_find_unused_parameters_true` を指定

### 8.3 LoRAターゲットモジュールによるボーカル消失（2025-01-17 解決）

**症状:**
- Fine-tuned モデルで推論すると、BGMは生成されるがボーカルが全く生成されない
- Epoch 36 から既に問題が発生（学習初期から）

**根本原因:**
SongBloomは2つのコンポーネントで構成されています:

| コンポーネント | 役割 | Attentionレイヤー |
|----------------|------|-------------------|
| AR Transformer (Llama) | スケッチトークン予測 | `q_proj`, `v_proj` |
| NAR DiT | 音声詳細生成（**ボーカル含む**） | `to_q`, `to_kv`, `to_qkv` |

NAR DiT のAttentionレイヤーを LoRA で変更すると、事前学習で獲得したボーカル生成パターンが破壊されます。

**解決方法:**
- LoRAターゲットを `q_proj,v_proj`（AR Transformerのみ）に変更
- `to_q`, `to_kv`, `to_qkv`（NAR DiT）は対象から除外

```bash
# 推奨（ARのみ）
--lora-target-modules "q_proj,v_proj"

# 非推奨（ボーカル生成が壊れる）
--lora-target-modules "q_proj,v_proj,to_q,to_kv,to_qkv"
```

## 9. 現状の課題

1. **LoRA + AR-only での再学習**
   - LoRAターゲットを `q_proj,v_proj` に変更して再学習が必要
   - NAR DiTを対象外にすることでボーカル生成が保持される見込み
2. **生成品質の評価**
   - 再学習後に推論テストが必要

## 10. 次にやるべきこと

1. **AR-only LoRAで再学習**
   - `--lora-target-modules "q_proj,v_proj"` で学習を実行
2. **推論テスト**
   - 10エポックごとにサンプル生成してボーカルの有無を確認
3. **追加データ検討**
   - 必要に応じてデータセット拡充

## 11. 参考ファイル
- 学習スクリプト: `train_japanese.py`
- 推論スクリプト: `infer.py`
- データ準備: `SongBloom/training/prepare_jacappella.py`
- VQ 学習: `SongBloom/training/train_vq.py`
- LoRA: `SongBloom/training/lora.py`

---

## 12. 外部データセット調査

### japanese-singing-voice (HuggingFace)
- URL: https://huggingface.co/datasets/tts-dataset/japanese-singing-voice
- 規模: 約1,000時間
- フォーマット: MP3（要変換）
- **歌詞データ: なし** → そのままでは使用不可
- ライセンス: CC-BY-NC-4.0

**結論**: 歌詞がないため、Whisper等での書き起こし + 構造タグ付与が必要。

### 必要データ量の目安
| 規模 | データ量 | 期待効果 |
|------|---------|---------|
| 最小 | 10-50時間 | 動作確認 |
| 小規模 | 100-500時間 | 実用レベル |
| 中規模 | 1,000-5,000時間 | 高品質 |

---

## 13. 推奨学習環境

### GPU環境
| 構成 | VRAM | 評価 |
|------|------|------|
| T4 × 4台 | 64GB | ✅ 十分 |
| A100 × 1台 | 80GB | ✅ 十分 |
| RTX 4090 × 1台 | 24GB | ⚠️ LoRA必須 |

### バッチサイズ推奨
| 学習モード | バッチサイズ/GPU | 勾配累積 | 実効バッチ |
|-----------|-----------------|---------|-----------|
| LoRA | 4 | 2 | 32 (4GPU) |
| 全パラメータ | 2 | 2 | 16 (4GPU) |

### ディスク容量
| データ規模 | 推奨容量 |
|-----------|---------|
| 10時間（実験） | 50GB |
| 100時間（小規模） | 150GB |
| 1,000時間（本格） | 1TB |
