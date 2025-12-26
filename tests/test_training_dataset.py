import json
import os
import tempfile
import unittest

import torch
from unittest import mock

from SongBloom.training.dataset import DatasetConfig, SongBloomTrainDataset, collate_training_batch


class FakeVAE:
    def __init__(self, channel_dim=4):
        self.channel_dim = channel_dim

    def encode(self, wav: torch.Tensor):
        # wav: [B, C, T]
        b, _, t = wav.shape
        return torch.zeros((b, self.channel_dim, t), dtype=wav.dtype)


class TestTrainingDataset(unittest.TestCase):
    def test_dataset_cache_and_collate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sr = 8000
            audio_path = os.path.join(tmpdir, "audio.wav")
            prompt_path = os.path.join(tmpdir, "prompt.wav")
            jsonl_path = os.path.join(tmpdir, "data.jsonl")
            cache_dir = os.path.join(tmpdir, "cache")

            item = {
                "idx": "sample_001",
                "audio_path": audio_path,
                "lyrics": "[verse] テスト. , [chorus] テスト.",
                "prompt_wav": prompt_path,
                "sketch_tokens": [0] * 80,
            }
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")

            cfg = DatasetConfig(
                jsonl_path=jsonl_path,
                sample_rate=sr,
                prompt_len=0.01,
                cache_dir=cache_dir,
                use_cache=True,
                process_lyrics=False,
            )
            dataset = SongBloomTrainDataset(cfg, vae=FakeVAE(), block_size=16)
            fake_audio = torch.randn(1, 80)
            fake_prompt = torch.randn(1, 40)
            with mock.patch.object(dataset, "_load_audio", return_value=fake_audio), \
                mock.patch.object(dataset, "_load_prompt", return_value=fake_prompt):
                sample = dataset[0]
            self.assertTrue(os.path.exists(os.path.join(cache_dir, "sample_001.pt")))

            # remove audio file to ensure cache is used
            with mock.patch.object(dataset, "_load_prompt", return_value=fake_prompt):
                cached_sample = dataset[0]
            self.assertEqual(sample["length"], cached_sample["length"])

            batch = collate_training_batch([sample, cached_sample])
            self.assertEqual(batch["audio_latent"].shape[0], 2)
            self.assertEqual(batch["sketch_tokens"].shape[0], 2)
            self.assertEqual(batch["prompt_wav"].shape[0], 2)

            # length should be multiple of block_size
            self.assertEqual(sample["length"] % 16, 0)


if __name__ == "__main__":
    unittest.main()
