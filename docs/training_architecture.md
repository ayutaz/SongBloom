# SongBloom 学習アーキテクチャ

本ドキュメントは論文 [arXiv:2506.07634](https://arxiv.org/abs/2506.07634) に基づき、SongBloomの学習実装に必要な情報をまとめたものです。

## 1. 概要

SongBloomは**自己回帰拡散モデル（Autoregressive Diffusion Model）**を用いたフルレングス楽曲生成フレームワークです。歌詞と10秒のリファレンス音声から最大150秒（4分）の楽曲を生成します。

### 主な特徴

- LM（言語モデル）とDiffusionを**統合した単一モデル**（カスケードではない）
- **インターリーブ生成**：スケッチ（意味）とアコースティック（音響）をパッチ単位で交互に生成
- 2Bパラメータ、48kHz出力

### モデル比較

| モデル | アーキテクチャ | サイズ | 最大長 | サンプルレート |
|--------|--------------|--------|--------|--------------|
| SongGen | LM | 1.1B | 30s | 16kHz |
| SongEditor | LM + Diff. | 0.7B + 1B | 120s | 44.1kHz |
| DiffRhythm | Diff. | 1.1B | 285s | 44.1kHz |
| YuE | LM + LM | 7B + 1B | 300s | 44.1kHz |
| **SongBloom-tiny** | LM & Diff. | 1.3B | 60s | 48kHz |
| **SongBloom-full** | LM & Diff. | 2B | 150s | 48kHz |

※ "+"はカスケード、"&"は統合モデルを示す

---

## 2. アーキテクチャ

### 2.1 全体構成

SongBloomは3つの主要モジュールで構成されます：

```
入力: 歌詞(phoneme) + リファレンス音声(10秒)
         ↓
┌─────────────────────────────────────┐
│  Autoregressive Transformer Decoder │  ← スケッチトークン生成
│  (Causal Attention, LLaMA-2ベース)  │
└─────────────────────────────────────┘
         ↓ hidden vector + sketch tokens
┌─────────────────────────────────────┐
│  Non-Autoregressive DiT             │  ← アコースティック潜在変数生成
│  (Full Attention, Rectified Flow)   │
└─────────────────────────────────────┘
         ↓ acoustic latents
┌─────────────────────────────────────┐
│  VAE Decoder (stable-audio-vae)     │  ← 波形復元
└─────────────────────────────────────┘
         ↓
出力: 48kHz 2ch 音声 (FLAC)
```

### 2.2 データ表現

#### 歌詞前処理

- **構造フラグ**: `[verse]`, `[chorus]`, `[bridge]`（ボーカル）、`[intro]`, `[inst]`, `[outro]`（非ボーカル）
- テキストを正規化後、**phoneme列**に変換
- 構造フラグはパラグラフの先頭に付与

#### スケッチトークン

- **MuQ**（SSL音楽モデル）から埋め込みを抽出
- 単一のVector Quantization層で離散化
- Codebookサイズ: 16384
- フレームレート: 25fps

#### アコースティック潜在変数

- **stable-audio-vae**で2ch 48kHz音声を圧縮
- 連続値の潜在変数シーケンス
- フレームレート: 25fps（スケッチと同期）

### 2.3 インターリーブ生成パラダイム

従来の2段階生成（全スケッチ→全アコースティック）ではなく、**パッチ単位で交互に生成**します：

```
パッチサイズ P = 16フレーム = 0.64秒

生成順序:
  sketch[0:P] → acoustic[0:P]
→ sketch[P:2P] → acoustic[P:2P]
→ sketch[2P:3P] → acoustic[2P:3P]
→ ...
```

**利点:**
1. スケッチ生成時に過去のアコースティック情報を活用可能
2. アコースティック生成時のシーケンス長を大幅に削減
3. 双方向の情報交換により一貫性が向上

**数式:**
```
p(a, s | C) = Π_{i=0}^{N} p_θ(s_{(iP:(i+1)P]} | s_{(0:iP]}, a_{(0:iP]}, C)
                        · p_φ(a_{(iP:(i+1)P]} | s_{(0:(i+1)P]}, a_{(0:iP]}, C)
```

---

## 3. モデル構成

### 3.1 ハイパーパラメータ

| パラメータ | SongBloom-tiny | SongBloom-full |
|-----------|----------------|----------------|
| 総パラメータ数 | 1.3B | 2B |
| 最大生成長 | 60秒 | 150秒 |
| AR Transformer層数 | 16 | 24 |
| NAR DiT層数 | 8 | 12 |
| Hidden dimension | 1536 | 1536 |
| Attention heads | 24 | 24 |
| パッチサイズ P | 16 | 16 |
| スケッチCodebook | 16384 | 16384 |
| フレームレート | 25fps | 25fps |

### 3.2 Transformer詳細

#### 自己回帰Transformer（ARステージ）

- **ベース**: LLaMA-2 decoderアーキテクチャ
- **Attention**: Causal attention（左から右への生成）
- **位置埋め込み**: RoPE（Rotary Position Embedding）
- **入力**: 条件（歌詞、スタイルプロンプト）+ スケッチトークン列
- **出力**: 次のスケッチトークン + hidden vector（各パッチ末尾）

#### 非自己回帰DiT（NARステージ）

- **ベース**: 同じTransformerアーキテクチャを修正
- **Attention**: Full attention（双方向）
- **位置埋め込み**: RoPE
- **入力**: hidden vector + スケッチトークン + 前パッチのアコースティック潜在変数
- **出力**: 現パッチのアコースティック潜在変数（ノイズ除去後）

#### アコースティックエンコーダ

- 2層の畳み込みネットワーク
- アコースティック潜在変数を圧縮してARステージに供給

### 3.3 重要な設計ポイント

1. **埋め込み共有**: スケッチトークンの埋め込み層はAR/NARステージで共有
2. **勾配伝播**: Diffusion損失からhidden vectorを経由してARステージへ逆伝播
3. **FlashAttention2互換**: カスタムattentionマスク不要の設計

---

## 4. 学習

### 4.1 損失関数

#### 総合損失

```
L = L_LM + λ · L_flow
```
- λ = 0.1（経験的に決定）

#### スケッチ生成損失（Cross-Entropy）

```
L_LM = -1/(NP) Σ_{i=0}^{N-1} Σ_{j=1}^{P} log p_θ(s_{iP+j} | s_{<iP+j}, a_{<iP}, C)
```
- N: パッチ数
- P: パッチサイズ（16）
- s_t: 時刻tのスケッチトークン

#### 拡散損失（Rectified Flow Matching）

```
L_flow = E_{t~π_t, z} [||v_φ(t, z_t | ·) - (z_1 - z_0)||²]
```

- t: logit-normal分布 π_t からサンプリング
- z_0: 元のアコースティック潜在変数 a_{(iP:(i+1)P]}
- z_1: ガウスノイズ N(0, I)
- v_φ: 速度場を予測するネットワーク

**Rectified Flow補間:**
```
z_t = (1 - t) · z_0 + t · z_1
```

### 4.2 学習設定

| 項目 | 値 |
|------|-----|
| データセット | 100K時間（中国語・英語楽曲） |
| オプティマイザ | AdamW |
| 学習率 | 1e-4 |
| スケジューラ | Cosine（warm-up 2000ステップ） |
| バッチサイズ | 128 |
| 学習ステップ | 約150K |
| GPU | 16 × A100 |
| 学習時間 | 約1週間 |
| 分散学習 | DeepSpeed ZeRO |

### 4.3 Classifier-Free Guidance (CFG)

学習時:
- hidden vectorとsketch tokensを統合した条件として扱う
- ランダムにjointでマスク（条件なし生成を学習）
- sketch tokensのみの追加マスクも適用（hidden vectorに情報を集約）

推論時:
- CFG係数: 1.5
- 条件付き/条件なし出力を補間

### 4.4 データ前処理パイプライン

```
元の楽曲データ
    ↓
┌─────────────────────────────────────┐
│ 1. Demucs: 音源分離                  │
│    → ボーカル / 伴奏トラック分離      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. WhisperX: 歌詞アライメント         │
│    → 不一致の歌詞をフィルタ           │
│    → 欠落単語の復元                   │
│    → タイムスタンプの精緻化           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. 構造解析 (Kim & Nam, 2023)        │
│    → verse/chorus/intro等の検出      │
└─────────────────────────────────────┘
    ↓
学習用データセット
```

---

## 5. 推論

### 5.1 推論パラメータ

| 項目 | 値 |
|------|-----|
| CFG係数 | 1.5 |
| Top-k サンプリング | k = 200 |
| Temperature | 0.9 |
| Diffusionステップ | 36 |
| ODEソルバー | Euler |
| RTF (Real-Time Factor) | 1.649 |

### 5.2 推論フロー

```
1. 歌詞をphoneme列に変換
2. リファレンス音声をVAEでエンコード
3. FOR i = 0 to N-1:
   a. ARステージ: sketch[iP:(i+1)P] と hidden vector を生成
   b. NARステージ: 36ステップのdiffusionで acoustic[iP:(i+1)P] を生成
4. 全アコースティック潜在変数をVAEでデコード
5. 48kHz音声を出力
```

---

## 6. 必要な外部モデル/ライブラリ

### 学習に必要

| モデル/ライブラリ | 用途 | 参照 |
|------------------|------|------|
| **MuQ** | スケッチトークン抽出 | [arXiv:2501.01108](https://arxiv.org/abs/2501.01108) |
| **stable-audio-vae** | 波形⇔潜在変数変換 | [GitHub](https://github.com/Stability-AI/stable-audio-tools) |
| **Demucs** | 音源分離 | [GitHub](https://github.com/facebookresearch/demucs) |
| **WhisperX** | 歌詞アライメント | [GitHub](https://github.com/m-bain/whisperX) |
| **構造解析器** | 楽曲構造検出 | Kim & Nam (WASPAA 2023) |
| **DeepSpeed** | 分散学習 | [GitHub](https://github.com/microsoft/DeepSpeed) |

### 推論に必要（既存実装で使用）

- MuQ（スケッチ抽出）
- stable-audio-vae（波形復元）
- G2P（歌詞→phoneme変換）

---

## 7. 実装チェックリスト

### Phase 1: データパイプライン
- [ ] 音源分離（Demucs統合）
- [ ] 歌詞アライメント（WhisperX統合）
- [ ] 構造解析器の統合
- [ ] スケッチトークン抽出（MuQ + VQ）
- [ ] アコースティック潜在変数抽出（stable-audio-vae）
- [ ] データローダー実装

### Phase 2: モデルアーキテクチャ
- [ ] AR Transformer（LLaMA-2ベース）
- [ ] NAR DiT（Rectified Flow）
- [ ] アコースティックエンコーダ
- [ ] 埋め込み共有機構
- [ ] インターリーブ生成ロジック

### Phase 3: 学習ループ
- [ ] 損失関数（L_LM + λ·L_flow）
- [ ] CFGマスキング
- [ ] DeepSpeed統合
- [ ] チェックポイント管理
- [ ] ログ/可視化

### Phase 4: 評価
- [ ] PER（Phoneme Error Rate）
- [ ] MCC（MuLan Cycle Consistency）
- [ ] FAD（Fréchet Audio Distance）
- [ ] SER（Structural Error Rate）

---

## 8. 参考文献

- [SongBloom論文](https://arxiv.org/abs/2506.07634)
- [DiTAR: Diffusion Transformer Autoregressive](https://arxiv.org/abs/2502.03930)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [LLaMA-2](https://arxiv.org/abs/2307.09288)
- [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
