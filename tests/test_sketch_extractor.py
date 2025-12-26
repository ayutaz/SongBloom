import os
import tempfile
import unittest

import numpy as np
import torch

from SongBloom.training.sketch import PrecomputedSketchExtractor


class TestPrecomputedSketchExtractor(unittest.TestCase):
    def test_load_pt_and_npy(self):
        tokens = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "sketch.pt")
            npy_path = os.path.join(tmpdir, "sketch.npy")
            torch.save(tokens, pt_path)
            np.save(npy_path, tokens.numpy())

            extractor = PrecomputedSketchExtractor()

            pt_loaded = extractor.extract_from_item({"sketch_path": pt_path})
            npy_loaded = extractor.extract_from_item({"sketch_path": npy_path})

            self.assertTrue(torch.equal(pt_loaded, tokens))
            self.assertTrue(torch.equal(npy_loaded, tokens))
            self.assertEqual(pt_loaded.dtype, torch.long)
            self.assertEqual(npy_loaded.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
