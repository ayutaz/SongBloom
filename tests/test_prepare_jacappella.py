import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

from SongBloom.training.prepare_jacappella import _build_lyrics, prepare_jacappella


def _write_wav(path: str, sr: int, seconds: float) -> None:
    import wave

    samples = int(sr * seconds)
    audio = np.zeros((samples,), dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())


def _write_musicxml(path: str, lyrics: str) -> None:
    score = ET.Element("score-partwise")
    part = ET.SubElement(score, "part")
    measure = ET.SubElement(part, "measure")
    note = ET.SubElement(measure, "note")
    lyric = ET.SubElement(note, "lyric")
    text = ET.SubElement(lyric, "text")
    text.text = lyrics
    tree = ET.ElementTree(score)
    tree.write(path, encoding="utf-8", xml_declaration=True)


class TestPrepareJaCappella(unittest.TestCase):
    def test_build_lyrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "song_SVS.musicxml")
            _write_musicxml(xml_path, "テスト。")
            lyrics = _build_lyrics(xml_path)
            self.assertTrue(lyrics.startswith("[verse]"))
            self.assertTrue(lyrics.endswith("."))

    def test_prepare_jacappella_no_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "jaCappella", "subset", "song")
            os.makedirs(root, exist_ok=True)
            audio_path = os.path.join(root, "lead_vocal.wav")
            xml_path = os.path.join(root, "song_SVS.musicxml")
            _write_wav(audio_path, sr=8000, seconds=1.0)
            _write_musicxml(xml_path, "テスト。ABC123")

            out_dir = os.path.join(tmpdir, "prepared")
            jsonl_path = prepare_jacappella(
                output_dir=out_dir,
                download_dir=os.path.join(tmpdir, "jaCappella"),
                audio_type="lead_vocal",
                musicxml_type="svs",
                subsets=None,
                prompt_sec=0.5,
                sample_rate=8000,
                overwrite=True,
                no_download=True,
                skip_empty_lyrics=True,
                clean_japanese=True,
            )

            self.assertTrue(os.path.exists(jsonl_path))
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("テスト", lines[0])
            self.assertNotIn("ABC", lines[0])


if __name__ == "__main__":
    unittest.main()
