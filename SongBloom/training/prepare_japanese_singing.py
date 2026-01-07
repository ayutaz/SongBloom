"""
Prepare japanese-singing-voice dataset from HuggingFace for SongBloom training.

This script downloads audio from the japanese-singing-voice dataset,
transcribes lyrics using OpenAI Whisper API, and generates training JSONL.

Dataset: https://huggingface.co/datasets/tts-dataset/japanese-singing-voice
"""

import argparse
import json
import os
import re
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torchaudio
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from openai import OpenAI

# Load environment variables from .env
load_dotenv()


@dataclass
class PrepareConfig:
    """Configuration for dataset preparation."""

    # Output settings
    output_dir: str = "./data/japanese_singing_prepared"
    download_dir: str = "./data/japanese_singing_raw"

    # Subset selection
    target_hours: float = 20.0
    min_score: int = 50
    max_duration_sec: int = 300  # 5 minutes
    min_duration_sec: int = 30

    # Audio settings
    sample_rate: int = 48000
    prompt_sec: float = 10.0

    # Whisper API settings
    whisper_model: str = "whisper-1"
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0

    # Processing flags
    dry_run: bool = False
    resume: bool = False
    overwrite: bool = False
    clean_japanese: bool = True

    # HuggingFace dataset
    hf_repo: str = "tts-dataset/japanese-singing-voice"
    total_shards: int = 78


def _clean_japanese_only(text: str) -> str:
    """Remove non-Japanese characters from lyrics."""
    # Keep Japanese characters (hiragana, katakana, kanji) and basic punctuation
    text = re.sub(r"[^ぁ-ゟ゠-ヿ一-鿿々ー・.,\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[,]{2,}", ",", text)
    return text.strip("., ")


def _split_by_length(text: str, chunk_size: int = 60) -> List[str]:
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


def _infer_sections(text: str, duration_sec: float) -> str:
    """
    Infer section structure from lyrics text.

    Heuristic approach:
    - Split by periods to get sentences
    - Group into verse/chorus sections
    - Add intro/outro based on duration
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
        if len(current_section) >= 2 or i == len(sentences) - 1:
            sections.append(". ".join(current_section) + ".")
            current_section = []

    if not sections:
        return f"[verse] {text}."

    # Build formatted lyrics with section tags
    formatted_parts = []

    # Add intro based on duration (roughly 1 tag per second for 150s model)
    intro_tags = min(int(duration_sec * 0.02), 5)  # ~2% of duration, max 5
    if intro_tags > 0:
        formatted_parts.append(" ".join(["[intro]"] * intro_tags))

    # Alternate between verse and chorus
    for i, section in enumerate(sections):
        if i % 2 == 0:
            formatted_parts.append(f"[verse] {section}")
        else:
            formatted_parts.append(f"[chorus] {section}")

    # Add outro
    outro_tags = min(int(duration_sec * 0.01), 3)  # ~1% of duration, max 3
    if outro_tags > 0:
        formatted_parts.append(" ".join(["[outro]"] * outro_tags))

    return " , ".join(formatted_parts)


def _download_shard(repo_id: str, shard_idx: int, download_dir: str) -> str:
    """Download a single TAR shard from HuggingFace."""
    filename = f"data/train-{shard_idx:04d}.tar"
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=download_dir,
        local_dir_use_symlinks=False,
    )
    return local_path


def _parse_tar_metadata(tar_path: str) -> List[Dict]:
    """Parse metadata from TAR shard without extracting audio."""
    samples = []
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".json"):
                f = tar.extractfile(member)
                if f:
                    try:
                        metadata = json.load(f)
                        metadata["_tar_path"] = tar_path
                        metadata["_json_name"] = member.name
                        metadata["_mp3_name"] = member.name.replace(".json", ".mp3")
                        samples.append(metadata)
                    except json.JSONDecodeError:
                        continue
    return samples


def _filter_samples(
    samples: List[Dict], config: PrepareConfig
) -> Tuple[List[Dict], float]:
    """Filter samples by quality and duration, return filtered list and total hours."""
    filtered = []
    total_seconds = 0

    for sample in samples:
        # Check quality score
        score = sample.get("dataset_score", 0)
        if score < config.min_score:
            continue

        # Check duration
        duration = sample.get("duration_sec", 0)
        if duration < config.min_duration_sec or duration > config.max_duration_sec:
            continue

        filtered.append(sample)
        total_seconds += duration

        # Stop if we've reached target hours
        if total_seconds / 3600 >= config.target_hours:
            break

    return filtered, total_seconds / 3600


def _extract_mp3_from_tar(tar_path: str, mp3_name: str, output_path: str) -> bool:
    """Extract a single MP3 file from TAR archive."""
    try:
        with tarfile.open(tar_path, "r") as tar:
            member = tar.getmember(mp3_name)
            with tar.extractfile(member) as src:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as dst:
                    dst.write(src.read())
        return True
    except Exception as e:
        print(f"Error extracting {mp3_name}: {e}")
        return False


def _convert_mp3_to_wav(mp3_path: str, wav_path: str, sample_rate: int = 48000) -> bool:
    """Convert MP3 to 48kHz stereo WAV."""
    try:
        wav, sr = torchaudio.load(mp3_path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        # Convert to stereo if mono
        if wav.ndim == 2 and wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0).repeat(2, 1)

        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        torchaudio.save(wav_path, wav, sample_rate)
        return True
    except Exception as e:
        print(f"Error converting {mp3_path}: {e}")
        return False


def _create_prompt(audio_path: str, prompt_path: str, prompt_sec: float, sample_rate: int) -> bool:
    """Create 10-second prompt from audio."""
    try:
        wav, sr = torchaudio.load(audio_path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        max_len = int(prompt_sec * sample_rate)
        wav = wav[..., :max_len]
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        torchaudio.save(prompt_path, wav, sample_rate)
        return True
    except Exception as e:
        print(f"Error creating prompt {prompt_path}: {e}")
        return False


def _transcribe_audio(
    audio_path: str,
    client: OpenAI,
    model: str = "whisper-1",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Optional[str]:
    """Transcribe audio using OpenAI Whisper API."""
    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language="ja",
                    response_format="text",
                )
            return transcript
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                print(f"Transcription error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed to transcribe {audio_path} after {max_retries} attempts: {e}")
                return None


def _validate_transcription(text: str, duration_sec: float) -> Tuple[bool, str]:
    """Validate transcription quality."""
    if not text or len(text.strip()) == 0:
        return False, "Empty transcription"

    # Check for Japanese characters
    japanese_chars = re.findall(r"[ぁ-ゟ゠-ヿ一-鿿]", text)
    if len(japanese_chars) < 5:
        return False, "Too few Japanese characters"

    # Check character density (roughly 2-5 chars per second for singing)
    char_density = len(text) / duration_sec
    if char_density < 0.5:
        return False, f"Character density too low ({char_density:.2f} chars/sec)"
    if char_density > 10:
        return False, f"Character density too high ({char_density:.2f} chars/sec)"

    # Check for excessive repetition
    if len(set(text)) < len(text) * 0.1:
        return False, "Excessive repetition detected"

    return True, ""


def _load_cached_transcription(cache_path: str) -> Optional[str]:
    """Load transcription from cache."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _save_cached_transcription(cache_path: str, text: str) -> None:
    """Save transcription to cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)


def prepare_japanese_singing(config: PrepareConfig) -> str:
    """
    Main preparation pipeline.

    Returns path to output JSONL.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.download_dir, exist_ok=True)

    audio_dir = os.path.join(config.output_dir, "audio")
    prompt_dir = os.path.join(config.output_dir, "prompts")
    cache_dir = os.path.join(config.output_dir, "transcriptions")
    jsonl_path = os.path.join(config.output_dir, "japanese_singing.jsonl")
    review_path = os.path.join(config.output_dir, "review_queue.jsonl")
    stats_path = os.path.join(config.output_dir, "statistics.json")

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not config.dry_run:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key) if api_key else None

    # Collect samples from shards
    print("Phase 1: Downloading and filtering samples...")
    all_samples = []
    accumulated_hours = 0

    for shard_idx in range(config.total_shards):
        if accumulated_hours >= config.target_hours:
            break

        print(f"  Processing shard {shard_idx}/{config.total_shards}...")
        try:
            tar_path = _download_shard(config.hf_repo, shard_idx, config.download_dir)
            shard_samples = _parse_tar_metadata(tar_path)
            filtered, hours = _filter_samples(shard_samples, config)

            # Only take what we need
            remaining_hours = config.target_hours - accumulated_hours
            selected = []
            selected_seconds = 0
            for sample in filtered:
                if selected_seconds / 3600 >= remaining_hours:
                    break
                selected.append(sample)
                selected_seconds += sample.get("duration_sec", 0)

            all_samples.extend(selected)
            accumulated_hours += selected_seconds / 3600
            print(f"    Found {len(selected)} samples, total: {accumulated_hours:.2f} hours")

        except Exception as e:
            print(f"    Error processing shard {shard_idx}: {e}")
            continue

    print(f"\nTotal samples selected: {len(all_samples)}")
    print(f"Total duration: {accumulated_hours:.2f} hours")

    # Cost estimation
    estimated_cost = accumulated_hours * 60 * 0.006  # $0.006 per minute
    print(f"Estimated Whisper API cost: ${estimated_cost:.2f}")

    if config.dry_run:
        print("\n[DRY RUN] Stopping before processing. No files modified.")
        return ""

    # Process samples
    print("\nPhase 2: Processing audio and transcribing...")
    items = []
    review_items = []
    processed = 0
    skipped = 0
    failed = 0

    for i, sample in enumerate(all_samples):
        sample_id = sample.get("id", f"unknown_{i}")
        duration_sec = sample.get("duration_sec", 0)
        print(f"  [{i + 1}/{len(all_samples)}] Processing {sample_id}...")

        # File paths
        mp3_path = os.path.join(config.download_dir, "temp", f"{sample_id}.mp3")
        wav_path = os.path.join(audio_dir, f"{sample_id}.wav")
        prompt_path = os.path.join(prompt_dir, f"{sample_id}_prompt.wav")
        cache_path = os.path.join(cache_dir, f"{sample_id}.txt")

        # Check if already processed
        if not config.overwrite and os.path.exists(wav_path) and os.path.exists(cache_path):
            transcription = _load_cached_transcription(cache_path)
            if transcription:
                print(f"    Using cached transcription")
            else:
                skipped += 1
                continue
        else:
            # Extract MP3
            tar_path = sample.get("_tar_path")
            mp3_name = sample.get("_mp3_name")
            if not tar_path or not mp3_name:
                failed += 1
                continue

            if not _extract_mp3_from_tar(tar_path, mp3_name, mp3_path):
                failed += 1
                continue

            # Convert to WAV
            if not _convert_mp3_to_wav(mp3_path, wav_path, config.sample_rate):
                failed += 1
                continue

            # Transcribe
            transcription = _load_cached_transcription(cache_path)
            if not transcription:
                transcription = _transcribe_audio(
                    mp3_path, client, config.whisper_model,
                    config.max_retries, config.retry_delay
                )
                if transcription:
                    _save_cached_transcription(cache_path, transcription)

            # Clean up temp MP3
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

        if not transcription:
            failed += 1
            continue

        # Validate transcription
        is_valid, reason = _validate_transcription(transcription, duration_sec)
        if not is_valid:
            review_items.append({
                "idx": sample_id,
                "audio_path": wav_path,
                "raw_transcription": transcription,
                "reason": reason,
                "metadata": sample,
            })
            print(f"    Flagged for review: {reason}")
            continue

        # Create prompt
        if not os.path.exists(prompt_path) or config.overwrite:
            _create_prompt(wav_path, prompt_path, config.prompt_sec, config.sample_rate)

        # Process lyrics
        if config.clean_japanese:
            transcription = _clean_japanese_only(transcription)

        lyrics = _infer_sections(transcription, duration_sec)

        # Build JSONL entry
        items.append({
            "idx": sample_id,
            "audio_path": wav_path,
            "lyrics": lyrics,
            "prompt_wav": prompt_path,
        })
        processed += 1

    # Write outputs
    print(f"\nPhase 3: Writing output files...")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} samples to {jsonl_path}")

    if review_items:
        with open(review_path, "w", encoding="utf-8") as f:
            for item in review_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(review_items)} samples to review queue")

    # Statistics
    stats = {
        "total_samples": len(all_samples),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "flagged_for_review": len(review_items),
        "total_hours": accumulated_hours,
        "estimated_cost": estimated_cost,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics saved to {stats_path}")

    print(f"\nDone! Processed {processed} samples.")
    return jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare japanese-singing-voice dataset for SongBloom training"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--download-dir", type=str, default="./data/japanese_singing_raw")
    parser.add_argument("--target-hours", type=float, default=20.0)
    parser.add_argument("--min-score", type=int, default=50)
    parser.add_argument("--min-duration", type=int, default=30)
    parser.add_argument("--max-duration", type=int, default=180)
    parser.add_argument("--whisper-model", type=str, default="whisper-1")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--prompt-sec", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--dry-run", action="store_true", help="Estimate costs only")
    parser.add_argument("--resume", action="store_true", help="Resume from cache")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean-japanese", action="store_true", default=True)
    args = parser.parse_args()

    config = PrepareConfig(
        output_dir=args.output_dir,
        download_dir=args.download_dir,
        target_hours=args.target_hours,
        min_score=args.min_score,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        prompt_sec=args.prompt_sec,
        sample_rate=args.sample_rate,
        dry_run=args.dry_run,
        resume=args.resume,
        overwrite=args.overwrite,
        clean_japanese=args.clean_japanese,
    )

    jsonl_path = prepare_japanese_singing(config)
    if jsonl_path:
        print(f"\nOutput JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
