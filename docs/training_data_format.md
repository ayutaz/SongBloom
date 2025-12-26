# 学習データフォーマット

SongBloomの日本語Fine-tuning学習に必要なデータフォーマットを説明します。

## 概要

学習には以下の3種類のデータが必要です：

1. **楽曲WAVファイル**: 学習対象の音声
2. **歌詞テキスト**: 構造タグ付きの歌詞
3. **プロンプトWAVファイル**: スタイル参照用の10秒音声

加えて、**スケッチトークン**が必要です（MuQ + VQ）。
このリポジトリでは以下のいずれかで用意します：

- `sketch_path` で事前計算済みトークンを渡す
- `--sketch-mode muq` で学習時に自動抽出する

## JSONLフォーマット

学習データはJSONL（JSON Lines）形式で準備します。各行が1つの学習サンプルです。

### 必須フィールド

```json
{
  "idx": "song_001",
  "audio_path": "/path/to/song.wav",
  "lyrics": "[intro] [intro] , [verse] 桜の花が咲いている. 春の風が吹く. , [chorus] 夢を見ている. , [outro] [outro]",
  "prompt_wav": "/path/to/prompt.wav"
}
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `idx` | string | 一意のサンプルID |
| `audio_path` | string | 楽曲WAVファイルのパス |
| `lyrics` | string | 構造タグ付き歌詞 |
| `prompt_wav` | string | 10秒プロンプトのパス |

### オプションフィールド

```json
{
  "sketch_path": "/path/to/sketch_tokens.pt"
}
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `sketch_path` | string | スケッチトークンのファイルパス（.pt/.npy/.npz） |

**注意**:
- `sketch_path` を指定しない場合は `--sketch-mode muq` を使用してください。
- `.pt` の場合は `{"sketch": tensor}` 形式でも読み込み可能です。

---

## 楽曲WAVファイル仕様

### 技術仕様

| 項目 | 要件 |
|------|------|
| サンプルレート | 48kHz |
| チャンネル | ステレオ（2ch）推奨、モノラル可 |
| ビット深度 | 16bit以上 |
| 最大長 | 150秒（songbloom_full_150s）<br>240秒（songbloom_full_240s） |
| フォーマット | WAV |

### 注意点

- モノラル音声は自動的にステレオに複製されます
- 最大長を超える音声は自動的にトランケートされます
- ブロックサイズ（16フレーム = 0.64秒）の倍数にパディングされます

---

## 歌詞フォーマット

### 構造タグ

歌詞は構造タグで区切られます：

| タグ | 種類 | 説明 |
|-----|------|------|
| `[intro]` | 非ボーカル | イントロ（楽器のみ） |
| `[outro]` | 非ボーカル | アウトロ（楽器のみ） |
| `[inst]` | 非ボーカル | 間奏（楽器のみ） |
| `[verse]` | ボーカル | Aメロ/Bメロ |
| `[chorus]` | ボーカル | サビ |
| `[bridge]` | ボーカル | ブリッジ/大サビ前 |

### 長さの対応

非ボーカルタグは繰り返し数で長さを制御します：

**150sモデル（songbloom_full_150s）**:
- 各タグ = 約1秒
- 例: `[intro] [intro] [intro]` = 約3秒

**240sモデル（songbloom_full_240s）**:
- 各タグ = 約5秒
- 例: `[intro] [intro] [intro]` = 約15秒

### 区切り文字

| 文字 | 用途 |
|-----|------|
| `.`（ピリオド） | 文の区切り |
| `,`（カンマ） | セクションの区切り |
| ` `（スペース） | 単語/タグの区切り |

### 歌詞例

```
[intro] [intro] [intro] ,
[verse] 桜の花が咲いている. 春の風が吹く. 遠くで鳥が鳴いてる. ,
[verse] 君との思い出が蘇る. あの日の約束. 忘れはしない. ,
[chorus] 夢を見ている. 君と歩く道. 終わらない物語. ,
[inst] [inst] ,
[verse] 季節は巡る. 時は流れて. また会える日を. ,
[chorus] 夢を見ている. 君と歩く道. 終わらない物語. ,
[outro] [outro] [outro]
```

### 日本語の書き方

- ひらがな、カタカナ、漢字が使用可能
- G2P（pyopenjtalk）で自動的に音素に変換されます
- 英語の歌詞も混在可能

---

## プロンプトWAVファイル仕様

### 技術仕様

| 項目 | 要件 |
|------|------|
| サンプルレート | 48kHz |
| チャンネル | ステレオ（2ch）推奨 |
| 長さ | **10秒**（固定） |
| フォーマット | WAV |

### 用途

プロンプトWAVは生成する楽曲のスタイル（音色、雰囲気）を決定します：

- 同じ歌手/アーティストの別の曲
- 同じジャンルの参考曲
- 学習対象曲の一部（最初の10秒など）

### 推奨プラクティス

1. **学習対象曲の一部を使用**:
   - 曲の最初の10秒を切り出す
   - ffmpeg例: `ffmpeg -i song.wav -ss 0 -t 10 prompt.wav`

2. **別の曲を使用**:
   - 同じアーティスト/スタイルの曲を選ぶ
   - 音質が同等以上のものを選ぶ

---

## サンプルデータの作成

### ディレクトリ構成例

```
data/
├── japanese_songs.jsonl
├── audio/
│   ├── song_001.wav
│   ├── song_002.wav
│   └── ...
└── prompts/
    ├── song_001_prompt.wav
    ├── song_002_prompt.wav
    └── ...
```

### JSONLの作成例

```bash
# 手動で作成
cat > data/japanese_songs.jsonl << 'EOF'
{"idx": "song_001", "audio_path": "data/audio/song_001.wav", "lyrics": "[intro] [intro] , [verse] 歌詞1. , [chorus] サビ1. , [outro]", "prompt_wav": "data/prompts/song_001_prompt.wav"}
{"idx": "song_002", "audio_path": "data/audio/song_002.wav", "lyrics": "[intro] , [verse] 歌詞2. , [chorus] サビ2. , [outro] [outro]", "prompt_wav": "data/prompts/song_002_prompt.wav"}
EOF
```

### プロンプト切り出しスクリプト

```bash
#!/bin/bash
# 各楽曲から最初の10秒を切り出してプロンプトを作成

for audio in data/audio/*.wav; do
    filename=$(basename "$audio" .wav)
    ffmpeg -i "$audio" -ss 0 -t 10 -ar 48000 -ac 2 "data/prompts/${filename}_prompt.wav"
done
```

---

## 前処理パイプライン

学習前に以下の前処理が自動的に実行されます：

```
楽曲WAV (48kHz)
    │
    ├─→ StableVAE.encode() ──→ x_latent (64, T)
    │   アコースティック潜在変数
    │
    └─→ MuQ + VQ ──────────→ x_sketch (T,)
        スケッチトークン（`sketch_path` がない場合のみ）

歌詞テキスト
    │
    └─→ G2P_Mix ──────────→ phoneme tokens
        日本語/中国語/英語対応
```

### キャッシュ

前処理結果は `cache/training/` にキャッシュされ、2回目以降の学習では再利用されます。

---

## データ量の目安

| 規模 | サンプル数 | 合計時間 | 用途 |
|------|-----------|---------|------|
| 最小 | 5-10曲 | 10-20分 | 動作確認 |
| 小規模 | 50-100曲 | 1-3時間 | 実験 |
| 中規模 | 500-1000曲 | 10-30時間 | 本格学習 |

### 注意事項

- 著作権に注意してください（学術目的のみ）
- 音質の低い音源は避けてください
- 歌詞と音声の対応が正確であることを確認してください

---

## トラブルシューティング

### 音声が読み込めない

```
Error: Failed to load audio file
```

- ファイルパスが正しいか確認
- WAVフォーマットであるか確認
- サンプルレートが48kHzであるか確認

### 歌詞パースエラー

```
Error: Invalid lyrics format
```

- 構造タグが正しい形式か確認（`[verse]`など）
- セクション区切り`,`が正しく配置されているか確認
- 文末に`.`があるか確認

### メモリ不足

- `batch-size`を1に減らす
- `--use-lora`オプションを使用する
- 短い楽曲から始める
