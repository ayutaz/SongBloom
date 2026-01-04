# 現状まとめ（2025-01-05 更新）

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
  - 24kHz resample → MuQ 埋め込み(1024) → VQ(codebook=16384)。
  - VAE latent と 25fps で長さ合わせ。
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
- VQ の学習は **CPU で実行**（MPS は複素数 dtype で失敗）。

コマンド例:
```bash
./.venv/bin/python -m SongBloom.training.train_vq \
  --data-jsonl data/jacappella_prepared_ja/jacappella.jsonl \
  --output-path data/vq/vq_16384.pt \
  --steps 2000 \
  --batch-size 1 \
  --max-duration 10 \
  --device cpu
```

生成物:
- `data/vq/vq_16384.pt`

## 6. 学習（微調整）
### 6.1 旧データでの学習
- 出力先: `checkpoints/jacappella_muq_lora/`
- 目的: baseline 動作確認

### 6.2 日本語クリーニング後の学習（現状）
- 出力先: `checkpoints/jacappella_muq_lora_ja/`
- 主要設定:
  - batch=1 / max_epochs=5 / val_split=0.1
  - `--sketch-mode muq` + `--muq-vq-path data/vq/vq_16384.pt`
  - `--use-lora`（rank=16, alpha=32）
- 学習済みチェックポイント:
  - `checkpoints/jacappella_muq_lora_ja/last.ckpt`

コマンド例:
```bash
WANDB_API_KEY=... ./.venv/bin/python train_japanese.py \
  --data-jsonl data/jacappella_prepared_ja/jacappella.jsonl \
  --val-split 0.1 \
  --output-dir ./checkpoints/jacappella_muq_lora_ja \
  --batch-size 1 \
  --max-epochs 5 \
  --device mps \
  --sketch-mode muq \
  --muq-vq-path data/vq/vq_16384.pt \
  --muq-device cpu \
  --require-vq-path \
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

## 8. 現状の問題点（重要）
1. **データ数が極端に少ない（26件）**
   - 日本語歌唱を学習するには圧倒的に不足。
2. **学習ステップ数が少なすぎる**
   - max_epochs=5 でも約120 step 程度しか回らない。
3. **LoRA 適用範囲が限定的**
   - q/v/to_q/to_kv/to_qkvのみで、音声表現の変化が小さい。
4. **歌詞が長文連結で構造が弱い**
   - MusicXML から抽出した歌詞は句読点や区切りが少なく、音響対応が難しい。
5. **ベースモデルが日本語歌唱に最適化されていない**
   - phoneme は日本語だが、ベースが英語寄りのため発音が英語風になりやすい。

## 9. 次にやるべきこと（優先順）
1. **データ増量（最優先）**
   - 日本語歌唱データの追加が必須。
2. **学習ステップの増加**
   - 5,000〜10,000 step 相当に増やす。
3. **LoRAの更新範囲拡大**
   - `--lora-train-all` を試す／rank増加。
4. **歌詞区切りの改善**
   - MusicXML からフレーズ境界を反映し、句読点を増やす。

## 10. 参考ファイル
- 学習スクリプト: `train_japanese.py`
- 推論スクリプト: `infer.py`
- データ準備: `SongBloom/training/prepare_jacappella.py`
- VQ 学習: `SongBloom/training/train_vq.py`
- LoRA: `SongBloom/training/lora.py`

---

## 11. 外部データセット調査

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

## 12. 推奨学習環境

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
