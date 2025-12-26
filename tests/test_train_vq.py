import os
import tempfile
import unittest

import torch

from SongBloom.training.train_vq import save_vq, train_vq


def _embedding_iter():
    while True:
        yield torch.randn(2, 4, 8)


class TestTrainVQ(unittest.TestCase):
    def test_train_and_save(self):
        vq = train_vq(
            embedding_iter=_embedding_iter(),
            embedding_dim=8,
            codebook_size=16,
            steps=2,
            lr=1e-3,
            decay=0.9,
            commitment_weight=1.0,
            device=torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vq.pt")
            save_vq(path, vq, {"test": True})
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
