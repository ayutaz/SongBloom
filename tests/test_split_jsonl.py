import json
import os
import tempfile
import unittest

from SongBloom.training.split_jsonl import load_jsonl, split_items, write_jsonl


class TestSplitJsonl(unittest.TestCase):
    def test_split_items_ratio(self):
        items = [{"idx": str(i)} for i in range(10)]
        train_items, val_items = split_items(items, val_ratio=0.2, seed=1234)
        self.assertEqual(len(train_items), 8)
        self.assertEqual(len(val_items), 2)
        train_ids = {item["idx"] for item in train_items}
        val_ids = {item["idx"] for item in val_items}
        self.assertTrue(train_ids.isdisjoint(val_ids))

    def test_write_and_load(self):
        items = [{"idx": "a"}, {"idx": "b"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            write_jsonl(path, items)
            loaded = load_jsonl(path)
            self.assertEqual(items, loaded)


if __name__ == "__main__":
    unittest.main()
