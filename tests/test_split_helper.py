import json
import os
import tempfile
import types
import unittest

from SongBloom.training.split_jsonl import load_jsonl
import train_japanese


class TestSplitHelper(unittest.TestCase):
    def test_maybe_split_jsonl_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            items = [{"idx": str(i)} for i in range(10)]
            with open(input_path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")

            args = types.SimpleNamespace(
                data_jsonl=input_path,
                val_jsonl=None,
                val_split=0.2,
                split_seed=1234,
                overwrite_split=True,
                output_dir=tmpdir,
                require_length_match=False,
                log_length_mismatch=False,
                max_mismatch_logs=0,
                verify_lengths=True,
            )

            train_path, val_path = train_japanese.maybe_split_jsonl(args)
            self.assertTrue(os.path.exists(train_path))
            self.assertTrue(os.path.exists(val_path))

            train_items = load_jsonl(train_path)
            val_items = load_jsonl(val_path)
            self.assertEqual(len(train_items), 8)
            self.assertEqual(len(val_items), 2)


if __name__ == "__main__":
    unittest.main()
