import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from SongBloom.training.dataset import DatasetConfig, SongBloomTrainDataset
from SongBloom.training.sketch import SketchExtractor


class FakeVAE:
    def __init__(self, channel_dim=4):
        self.channel_dim = channel_dim

    def encode(self, wav: torch.Tensor):
        b, _, t = wav.shape
        return torch.zeros((b, self.channel_dim, t), dtype=wav.dtype)


class DummyExtractor(SketchExtractor):
    def __init__(self):
        self.called = None

    def extract(self, wav: torch.Tensor, sample_rate: int, target_length=None) -> torch.Tensor:
        self.called = {
            "shape": tuple(wav.shape),
            "sample_rate": sample_rate,
            "target_length": target_length,
        }
        if target_length is None:
            target_length = wav.shape[-1]
        return torch.arange(target_length, dtype=torch.long)


class TestExternalSketchExtractorPath(unittest.TestCase):
    def test_dataset_uses_external_extractor_with_target_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "data.jsonl")
            item = {
                "idx": "sample_001",
                "audio_path": "dummy.wav",
                "lyrics": "[verse] テスト. , [chorus] テスト.",
                "prompt_wav": "prompt.wav",
            }
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")

            cfg = DatasetConfig(
                jsonl_path=jsonl_path,
                sample_rate=8000,
                prompt_len=0.01,
                process_lyrics=False,
            )

            extractor = DummyExtractor()
            dataset = SongBloomTrainDataset(cfg, vae=FakeVAE(), block_size=16, sketch_extractor=extractor)

            fake_audio = torch.randn(1, 64)
            fake_prompt = torch.randn(1, 40)
            with mock.patch.object(dataset, "_load_audio", return_value=fake_audio), \
                mock.patch.object(dataset, "_load_prompt", return_value=fake_prompt):
                sample = dataset[0]

            self.assertIsNotNone(extractor.called)
            self.assertEqual(extractor.called["target_length"], 64)
            self.assertEqual(sample["length"], 64)
            self.assertEqual(sample["sketch_tokens"].shape[0], 64)


if __name__ == "__main__":
    unittest.main()
