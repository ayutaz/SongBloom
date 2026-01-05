"""Prepare Tohoku Kiritan singing database for SongBloom training.

Download the database manually from https://zunko.jp/kiridev/login.php
and extract to data/kiritan_raw/ before running this script.

Expected directory structure:
    data/kiritan_raw/
    ├── wav/           # 50 singing WAV files (96kHz)
    ├── mono_label/    # Phoneme labels
    └── musicxml/      # MusicXML files with lyrics

Usage:
    uv run python -m SongBloom.training.prepare_kiritan \
        --input-dir data/kiritan_raw \
        --output-dir data/kiritan_prepared \
        --prompt-sec 10 \
        --clean-japanese
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import torchaudio


def _find_wav_files(wav_dir: str) -> List[str]:
    """Find all WAV files in the wav directory."""
    if not os.path.isdir(wav_dir):
        return []
    return sorted([
        os.path.join(wav_dir, f)
        for f in os.listdir(wav_dir)
        if f.endswith(".wav")
    ])


def _find_musicxml_for_wav(wav_path: str, musicxml_dir: str) -> Optional[str]:
    """Find matching MusicXML file for a WAV file."""
    wav_name = os.path.splitext(os.path.basename(wav_path))[0]

    # Try exact match first
    for ext in [".musicxml", ".xml"]:
        xml_path = os.path.join(musicxml_dir, wav_name + ext)
        if os.path.exists(xml_path):
            return xml_path

    # Try partial match (e.g., "01_song" matches "01_song.musicxml")
    for f in os.listdir(musicxml_dir):
        if f.startswith(wav_name) and (f.endswith(".musicxml") or f.endswith(".xml")):
            return os.path.join(musicxml_dir, f)

    return None


def _extract_lyrics_from_musicxml(xml_path: str) -> str:
    """Extract lyrics from MusicXML file."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return ""

    root = tree.getroot()
    texts: List[str] = []

    # Handle namespace if present
    ns = {"": ""}
    if root.tag.startswith("{"):
        ns_end = root.tag.find("}")
        ns[""] = root.tag[1:ns_end]

    # Find all lyric elements
    for lyric in root.iter():
        if lyric.tag.endswith("lyric") or "lyric" in lyric.tag:
            for text_node in lyric.iter():
                if (text_node.tag.endswith("text") or "text" in text_node.tag) and text_node.text:
                    text = text_node.text.strip()
                    if text and text not in ["_", "-", " "]:
                        texts.append(text)

    raw = "".join(texts)
    # Normalize
    raw = raw.replace("\u3000", "")  # Full-width space
    raw = raw.replace("\n", "")
    raw = raw.replace("。", ".")
    raw = raw.replace("、", ".")
    raw = re.sub(r"\.+", ".", raw)
    return raw.strip(".")


def _clean_japanese_only(text: str) -> str:
    """Keep only Japanese characters and basic punctuation."""
    # Keep hiragana, katakana, kanji, and punctuation
    text = re.sub(r"[^ぁ-ゟ゠-ヿ一-鿿々ー・.,\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[,]{2,}", ",", text)
    return text.strip("., ")


def _split_by_length(text: str, chunk_size: int = 60) -> list[str]:
    """Split text into chunks of approximately chunk_size characters."""
    chunks = []
    current = ""
    for char in text:
        current += char
        if len(current) >= chunk_size:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _infer_sections(text: str) -> str:
    """Infer section structure from lyrics text.

    Heuristic approach:
    - Split by periods to get sentences
    - If no periods or only 1-2 sentences, split by character count
    - Group into verse/chorus sections
    - Add intro/outro based on number of sections
    """
    # Split by period, keeping non-empty parts
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    # If few sentences but long text, split by character count
    total_len = len(text)
    if len(sentences) <= 2 and total_len > 100:
        sentences = _split_by_length(text.replace(".", ""), chunk_size=60)

    if not sentences or len(sentences) <= 1:
        return f"[verse] {text}."

    # Group sentences into sections (roughly 2-3 sentences per section)
    sections = []
    current_section = []
    for i, sentence in enumerate(sentences):
        current_section.append(sentence)
        # Create new section every 2-3 sentences
        if len(current_section) >= 2 or i == len(sentences) - 1:
            sections.append(". ".join(current_section) + ".")
            current_section = []

    if not sections:
        return f"[verse] {text}."

    # Build formatted lyrics with section tags
    formatted_parts = []

    # Add intro (3 seconds worth of intro tags)
    formatted_parts.append("[intro] [intro] [intro]")

    # Alternate between verse and chorus
    for i, section in enumerate(sections):
        if i % 2 == 0:
            formatted_parts.append(f"[verse] {section}")
        else:
            formatted_parts.append(f"[chorus] {section}")

    # Add outro (2 seconds)
    formatted_parts.append("[outro] [outro]")

    return " , ".join(formatted_parts)


def _build_lyrics(xml_path: str, clean_japanese: bool) -> Optional[str]:
    """Build formatted lyrics from MusicXML."""
    text = _extract_lyrics_from_musicxml(xml_path)
    if not text:
        return None

    if clean_japanese:
        text = _clean_japanese_only(text)
        if not text:
            return None

    return _infer_sections(text)


def _make_prompt(
    audio_path: str,
    prompt_path: str,
    prompt_sec: float,
    target_sample_rate: int,
) -> None:
    """Create a prompt audio file from the beginning of the song."""
    wav, sr = torchaudio.load(audio_path)

    # Resample if needed (Kiritan is 96kHz, we need 48kHz)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)

    # Convert to mono if stereo
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Ensure mono dimension
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    # Take first N seconds
    max_len = int(prompt_sec * target_sample_rate)
    wav = wav[..., :max_len]

    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
    torchaudio.save(prompt_path, wav, target_sample_rate)


def prepare_kiritan(
    input_dir: str,
    output_dir: str,
    prompt_sec: float,
    sample_rate: int,
    overwrite: bool,
    skip_empty_lyrics: bool,
    clean_japanese: bool,
) -> str:
    """Prepare Kiritan dataset for SongBloom training."""
    wav_dir = os.path.join(input_dir, "wav")
    musicxml_dir = os.path.join(input_dir, "musicxml")

    if not os.path.isdir(wav_dir):
        raise RuntimeError(f"WAV directory not found: {wav_dir}")
    if not os.path.isdir(musicxml_dir):
        raise RuntimeError(f"MusicXML directory not found: {musicxml_dir}")

    wav_files = _find_wav_files(wav_dir)
    if not wav_files:
        raise RuntimeError(f"No WAV files found in {wav_dir}")

    os.makedirs(output_dir, exist_ok=True)
    prompt_dir = os.path.join(output_dir, "prompts")
    jsonl_path = os.path.join(output_dir, "kiritan.jsonl")

    items = []
    skipped = 0
    no_xml = 0

    for wav_path in wav_files:
        wav_name = os.path.splitext(os.path.basename(wav_path))[0]
        idx = f"kiritan_{wav_name}"

        # Find matching MusicXML
        xml_path = _find_musicxml_for_wav(wav_path, musicxml_dir)
        if xml_path is None:
            no_xml += 1
            if skip_empty_lyrics:
                continue
            lyrics = "[verse]"
        else:
            lyrics = _build_lyrics(xml_path, clean_japanese)
            if lyrics is None:
                if skip_empty_lyrics:
                    skipped += 1
                    continue
                lyrics = "[verse]"

        # Create prompt
        prompt_path = os.path.join(prompt_dir, f"{idx}_prompt.wav")
        if overwrite or not os.path.exists(prompt_path):
            try:
                _make_prompt(wav_path, prompt_path, prompt_sec, sample_rate)
            except Exception as e:
                print(f"Warning: Failed to create prompt for {wav_name}: {e}")
                continue

        items.append({
            "idx": idx,
            "audio_path": wav_path,
            "lyrics": lyrics,
            "prompt_wav": prompt_path,
        })

    # Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processed {len(items)} songs")
    if skipped:
        print(f"Skipped {skipped} files due to empty/invalid lyrics")
    if no_xml:
        print(f"Warning: {no_xml} WAV files had no matching MusicXML")

    return jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Tohoku Kiritan singing database for SongBloom training"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing kiritan data (with wav/, musicxml/ subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--prompt-sec",
        type=float,
        default=10.0,
        help="Prompt duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Target sample rate (default: 48000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prompt files",
    )
    parser.add_argument(
        "--skip-empty-lyrics",
        action="store_true",
        default=True,
        help="Skip files with empty lyrics (default: True)",
    )
    parser.add_argument(
        "--no-skip-empty-lyrics",
        action="store_false",
        dest="skip_empty_lyrics",
        help="Include files even with empty lyrics",
    )
    parser.add_argument(
        "--clean-japanese",
        action="store_true",
        help="Remove non-Japanese characters from lyrics",
    )

    args = parser.parse_args()

    jsonl_path = prepare_kiritan(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt_sec=args.prompt_sec,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
        skip_empty_lyrics=args.skip_empty_lyrics,
        clean_japanese=args.clean_japanese,
    )
    print(f"Wrote JSONL -> {jsonl_path}")


if __name__ == "__main__":
    main()
