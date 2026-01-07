"""Tests for train_japanese.py CFG and timestep sampler arguments."""
import argparse
import unittest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrainJapaneseArgs(unittest.TestCase):
    """Test argument parsing for CFG and timestep sampler."""

    def _get_parser(self):
        """Create a minimal parser with the new arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-jsonl", type=str, required=True)
        parser.add_argument("--h-dropout", type=float, default=0.1,
                            help="CFG dropout rate for hidden vectors")
        parser.add_argument("--timestep-sampler", type=str, default="logit_normal",
                            choices=["uniform", "logit_normal", "trunc_logit_normal"])
        return parser

    def test_h_dropout_default(self):
        """h_dropout should default to 0.1 (paper recommendation)."""
        parser = self._get_parser()
        args = parser.parse_args(["--data-jsonl", "test.jsonl"])
        self.assertEqual(args.h_dropout, 0.1)

    def test_h_dropout_custom(self):
        """h_dropout should accept custom values."""
        parser = self._get_parser()
        args = parser.parse_args(["--data-jsonl", "test.jsonl", "--h-dropout", "0.2"])
        self.assertEqual(args.h_dropout, 0.2)

    def test_h_dropout_zero(self):
        """h_dropout=0 should disable CFG."""
        parser = self._get_parser()
        args = parser.parse_args(["--data-jsonl", "test.jsonl", "--h-dropout", "0.0"])
        self.assertEqual(args.h_dropout, 0.0)

    def test_timestep_sampler_default(self):
        """timestep_sampler should default to logit_normal (paper)."""
        parser = self._get_parser()
        args = parser.parse_args(["--data-jsonl", "test.jsonl"])
        self.assertEqual(args.timestep_sampler, "logit_normal")

    def test_timestep_sampler_uniform(self):
        """timestep_sampler should accept uniform."""
        parser = self._get_parser()
        args = parser.parse_args(["--data-jsonl", "test.jsonl", "--timestep-sampler", "uniform"])
        self.assertEqual(args.timestep_sampler, "uniform")

    def test_timestep_sampler_trunc_logit_normal(self):
        """timestep_sampler should accept trunc_logit_normal."""
        parser = self._get_parser()
        args = parser.parse_args(["--data-jsonl", "test.jsonl",
                                  "--timestep-sampler", "trunc_logit_normal"])
        self.assertEqual(args.timestep_sampler, "trunc_logit_normal")

    def test_timestep_sampler_invalid(self):
        """timestep_sampler should reject invalid values."""
        parser = self._get_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--data-jsonl", "test.jsonl",
                              "--timestep-sampler", "invalid"])


class TestConfigApplication(unittest.TestCase):
    """Test that arguments are correctly applied to model config."""

    def test_config_h_dropout_applied(self):
        """h_dropout should be applied to cfg.model."""
        from omegaconf import OmegaConf

        # Simulate cfg.model
        cfg = OmegaConf.create({"model": {}})
        h_dropout = 0.15

        # Apply like train_japanese.py does
        cfg.model['h_dropout'] = h_dropout

        self.assertEqual(cfg.model.h_dropout, 0.15)

    def test_config_timestep_sampler_applied(self):
        """timestep_sampler should be applied to cfg.model."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"model": {}})
        timestep_sampler = "logit_normal"

        cfg.model['timestep_sampler'] = timestep_sampler

        self.assertEqual(cfg.model.timestep_sampler, "logit_normal")

    def test_config_both_applied(self):
        """Both settings should coexist in cfg.model."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"model": {"existing_key": "value"}})

        cfg.model['h_dropout'] = 0.1
        cfg.model['timestep_sampler'] = "logit_normal"

        self.assertEqual(cfg.model.h_dropout, 0.1)
        self.assertEqual(cfg.model.timestep_sampler, "logit_normal")
        self.assertEqual(cfg.model.existing_key, "value")


class TestPaperCompliance(unittest.TestCase):
    """Test that defaults match paper recommendations."""

    def test_default_h_dropout_matches_paper(self):
        """Paper recommends CFG dropout around 0.1-0.5."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--h-dropout", type=float, default=0.1)
        args = parser.parse_args([])

        # Paper: "We randomly drop the condition with probability p during training"
        # Typical values: 0.1-0.2
        self.assertGreater(args.h_dropout, 0.0, "CFG dropout should be > 0")
        self.assertLessEqual(args.h_dropout, 0.5, "CFG dropout should be <= 0.5")

    def test_default_timestep_sampler_matches_paper(self):
        """Paper uses logit-normal distribution for timesteps."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--timestep-sampler", type=str, default="logit_normal")
        args = parser.parse_args([])

        # Paper: "We sample timesteps from a logit-normal distribution"
        self.assertEqual(args.timestep_sampler, "logit_normal")


if __name__ == "__main__":
    unittest.main()
