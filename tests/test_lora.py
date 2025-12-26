import unittest

import torch
from torch import nn

from SongBloom.training.lora import LoRALinear, inject_lora, set_lora_trainable


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.linear(x))


class TestLoRA(unittest.TestCase):
    def test_inject_lora_and_trainable(self):
        model = ToyModel()
        replaced = inject_lora(
            model,
            target_modules=["linear", "block.0"],
            r=4,
            alpha=8,
            dropout=0.0,
        )
        self.assertIn("linear", replaced)
        self.assertIn("block.0", replaced)
        self.assertIsInstance(model.linear, LoRALinear)
        self.assertIsInstance(model.block[0], LoRALinear)

        set_lora_trainable(model, train_all=False)
        for name, param in model.named_parameters():
            if "lora_" in name:
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)

        x = torch.randn(2, 4)
        y = model(x)
        self.assertEqual(y.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
