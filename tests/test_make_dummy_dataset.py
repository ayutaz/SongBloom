import os
import tempfile
import unittest
import wave

import numpy as np

from SongBloom.training.make_dummy_dataset import make_dummy_dataset


class TestMakeDummyDataset(unittest.TestCase):
    def test_make_dummy_dataset_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            items = make_dummy_dataset(
                output_dir=tmpdir,
                num_samples=2,
                duration_sec=1.0,
                sample_rate=8000,
                prompt_sec=0.5,
                sketch_fps=25,
                codebook_size=128,
                sketch_ext=".npy",
                seed=1234,
            )

            self.assertEqual(len(items), 2)
            first = items[0]
            self.assertTrue(os.path.exists(first["audio_path"]))
            self.assertTrue(os.path.exists(first["prompt_wav"]))
            self.assertTrue(os.path.exists(first["sketch_path"]))

            with wave.open(first["audio_path"], "rb") as wf:
                self.assertEqual(wf.getframerate(), 8000)
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getnframes(), 8000)

            tokens = np.load(first["sketch_path"])
            self.assertEqual(tokens.shape[0], 25)


if __name__ == "__main__":
    unittest.main()
