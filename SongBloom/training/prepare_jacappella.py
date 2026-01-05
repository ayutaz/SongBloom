import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Iterable, List, Optional
import torchaudio
from huggingface_hub import snapshot_download


def _detect_root(local_dir: str) -> str:
    candidate = os.path.join(local_dir, "jaCappella")
    return candidate if os.path.isdir(candidate) else local_dir


def _download_dataset(local_dir: str, audio_type: str, musicxml_type: str) -> str:
    if audio_type == "lead_vocal":
        audio_pattern = "**/lead_vocal.wav"
    elif audio_type == "mixture":
        audio_pattern = "**/mixture.wav"
    elif audio_type == "mixture_stereo":
        audio_pattern = "**/mixture_stereo.wav"
    else:
        raise ValueError(f"Unknown audio_type: {audio_type}")

    if musicxml_type == "svs":
        xml_pattern = "**/*_SVS.musicxml"
    elif musicxml_type == "romaji":
        xml_pattern = "**/*_romaji.musicxml"
    elif musicxml_type == "default":
        xml_pattern = "**/*.musicxml"
    else:
        raise ValueError(f"Unknown musicxml_type: {musicxml_type}")

    snapshot_download(
        repo_id="jaCappella/jaCappella",
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[audio_pattern, xml_pattern, "**/meta.csv"],
    )

    return _detect_root(local_dir)


def _iter_musicxml(root_dir: str, subset_filters: Optional[Iterable[str]], musicxml_type: str) -> List[str]:
    if musicxml_type == "svs":
        suffix = "_SVS.musicxml"
    elif musicxml_type == "romaji":
        suffix = "_romaji.musicxml"
    else:
        suffix = ".musicxml"

    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if subset_filters:
            parts = dirpath.split(os.sep)
            if len(parts) >= 2:
                subset = parts[-2]
                if subset not in subset_filters:
                    continue
        for name in filenames:
            if name.endswith(suffix):
                paths.append(os.path.join(dirpath, name))
    return paths


def _extract_lyrics(xml_path: str) -> str:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return ""
    root = tree.getroot()
    texts: List[str] = []
    for lyric in root.iter():
        if not lyric.tag.endswith("lyric"):
            continue
        for text_node in lyric.iter():
            if text_node.tag.endswith("text") and text_node.text:
                text = text_node.text.strip()
                if text:
                    texts.append(text)

    raw = "".join(texts)
    raw = raw.replace("\u3000", "")
    raw = raw.replace("\n", "")
    raw = raw.replace("。", ".")
    raw = raw.replace("、", ".")
    raw = re.sub(r"\.+", ".", raw)
    return raw.strip(".")


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


def _build_lyrics(xml_path: str) -> Optional[str]:
    text = _extract_lyrics(xml_path)
    if not text:
        return None
    return _infer_sections(text)


def _clean_japanese_only(text: str) -> str:
    # Keep Japanese characters and basic punctuation for structure.
    text = re.sub(r"[^ぁ-ゟ゠-ヿ一-鿿々ー・.,\\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[,]{2,}", ",", text)
    return text.strip("., ")


def _make_prompt(audio_path: str, prompt_path: str, prompt_sec: float, sample_rate: int) -> None:
    wav, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    max_len = int(prompt_sec * sample_rate)
    wav = wav[..., :max_len]
    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
    torchaudio.save(prompt_path, wav, sample_rate)


def _infer_song_dir(xml_path: str) -> str:
    return os.path.dirname(xml_path)


def _infer_subset(root_dir: str, song_dir: str) -> Optional[str]:
    rel = os.path.relpath(song_dir, root_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        return parts[0]
    return None


def _infer_title(song_dir: str) -> str:
    return os.path.basename(song_dir)


def _audio_path(song_dir: str, audio_type: str) -> str:
    if audio_type == "lead_vocal":
        name = "lead_vocal.wav"
    elif audio_type == "mixture":
        name = "mixture.wav"
    else:
        name = "mixture_stereo.wav"
    return os.path.join(song_dir, name)


def prepare_jacappella(
    output_dir: str,
    download_dir: str,
    audio_type: str,
    musicxml_type: str,
    subsets: Optional[List[str]],
    prompt_sec: float,
    sample_rate: int,
    overwrite: bool,
    no_download: bool,
    skip_empty_lyrics: bool,
    clean_japanese: bool,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if not no_download:
        dataset_root = _download_dataset(download_dir, audio_type, musicxml_type)
    else:
        dataset_root = _detect_root(download_dir)

    xml_paths = _iter_musicxml(dataset_root, subsets, musicxml_type)
    if not xml_paths:
        raise RuntimeError("No MusicXML files found. Check dataset path or subset filters.")

    prompt_dir = os.path.join(output_dir, "prompts")
    jsonl_path = os.path.join(output_dir, "jacappella.jsonl")

    items = []
    skipped = 0
    for xml_path in xml_paths:
        song_dir = _infer_song_dir(xml_path)
        subset = _infer_subset(dataset_root, song_dir) or "unknown"
        title = _infer_title(song_dir)
        idx = f"{subset}_{title}"

        audio_path = _audio_path(song_dir, audio_type)
        if not os.path.exists(audio_path):
            continue

        prompt_path = os.path.join(prompt_dir, f"{idx}_prompt.wav")
        if overwrite or not os.path.exists(prompt_path):
            _make_prompt(audio_path, prompt_path, prompt_sec, sample_rate)

        lyrics = _build_lyrics(xml_path)
        if lyrics is not None and clean_japanese:
            parts = lyrics.split(" ", 1)
            if len(parts) == 2:
                tag, content = parts
                content = _clean_japanese_only(content)
                lyrics = f"{tag} {content}." if content else None
            else:
                lyrics = _clean_japanese_only(lyrics)
        if lyrics is None and skip_empty_lyrics:
            skipped += 1
            continue
        if lyrics is None:
            lyrics = "[verse]"
        items.append(
            {
                "idx": idx,
                "audio_path": audio_path,
                "lyrics": lyrics,
                "prompt_wav": prompt_path,
            }
        )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if skipped:
        print(f"skipped {skipped} files due to empty/invalid lyrics")
    return jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--download-dir", type=str, default="./data/jacappella")
    parser.add_argument("--audio-type", type=str, default="lead_vocal", choices=["lead_vocal", "mixture", "mixture_stereo"])
    parser.add_argument("--musicxml-type", type=str, default="svs", choices=["svs", "romaji", "default"])
    parser.add_argument("--subsets", type=str, default=None, help="Comma-separated subset names")
    parser.add_argument("--prompt-sec", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-empty-lyrics", action="store_true", default=True)
    parser.add_argument("--clean-japanese", action="store_true", help="Remove non-Japanese characters from lyrics")
    args = parser.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()] if args.subsets else None
    jsonl_path = prepare_jacappella(
        output_dir=args.output_dir,
        download_dir=args.download_dir,
        audio_type=args.audio_type,
        musicxml_type=args.musicxml_type,
        subsets=subsets,
        prompt_sec=args.prompt_sec,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
        no_download=args.no_download,
        skip_empty_lyrics=args.skip_empty_lyrics,
        clean_japanese=args.clean_japanese,
    )
    print(f"wrote jsonl -> {jsonl_path}")


if __name__ == "__main__":
    main()
