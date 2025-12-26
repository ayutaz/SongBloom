"""
Japanese G2P (Grapheme-to-Phoneme) module using pyopenjtalk-plus.
"""

import re
import pyopenjtalk

from .symbols import punctuation


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "「": "",
    "」": "",
    "『": "",
    "』": "",
    "【": "",
    "】": "",
    "（": "",
    "）": "",
    "〜": "-",
    "～": "-",
    "ー": "-",
    "　": " ",
}


def replace_punctuation(text: str) -> str:
    """Replace Japanese punctuation with standard punctuation."""
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    # Keep only Japanese characters (hiragana, katakana, kanji) and standard punctuation
    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3005-\u3007"
        + "".join(punctuation)
        + r"a-zA-Z0-9\s]+",
        "",
        replaced_text,
    )
    return replaced_text


def text_normalize(text: str) -> str:
    """Normalize Japanese text."""
    return replace_punctuation(text)


def g2p(text: str) -> list[str]:
    """
    Convert Japanese text to phoneme list.

    Uses pyopenjtalk to convert text to phonemes.
    Returns a list of phonemes compatible with ja_symbols in symbols.py.
    """
    phones = []

    # Split by punctuation while keeping them
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [s for s in re.split(pattern, text) if s.strip()]

    for sentence in sentences:
        # Check if it's just punctuation
        if sentence in punctuation:
            phones.append(sentence)
            continue

        # Get phonemes from pyopenjtalk
        # pyopenjtalk.g2p returns space-separated phoneme string
        try:
            phoneme_str = pyopenjtalk.g2p(sentence, kana=False)
            if phoneme_str:
                # Split phonemes and filter
                for ph in phoneme_str.split():
                    # pyopenjtalk outputs phonemes like: k o N n i ch i w a
                    # These should match ja_symbols in symbols.py
                    if ph and ph not in ["pau", "sil"]:  # Skip pause/silence markers
                        phones.append(ph)
        except Exception:
            # If conversion fails, skip this segment
            pass

    return phones


if __name__ == "__main__":
    # Test
    test_texts = [
        "こんにちは",
        "今日はいい天気ですね。",
        "私の名前は花子です！",
        "桜の花が咲いている。",
    ]

    for text in test_texts:
        normalized = text_normalize(text)
        phonemes = g2p(normalized)
        print(f"Text: {text}")
        print(f"Normalized: {normalized}")
        print(f"Phonemes: {phonemes}")
        print()
