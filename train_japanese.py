import argparse
import os
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, DictConfig
from huggingface_hub import hf_hub_download

from SongBloom.models.songbloom.songbloom_pl import SongBloom_PL
from SongBloom.models.vae_frontend import StableVAE
from SongBloom.training.dataset import DatasetConfig, SongBloomTrainDataset, collate_training_batch
from SongBloom.training.sketch import ExternalSketchExtractor
from SongBloom.training.split_jsonl import split_items, load_jsonl, write_jsonl

NAME2REPO = {
    "songbloom_full_150s": "CypressYang/SongBloom",
    "songbloom_full_150s_dpo": "CypressYang/SongBloom",
    "songbloom_full_240s": "CypressYang/SongBloom_long",
}


def hf_download(model_name="songbloom_full_150s", local_dir="./cache", **kwargs):
    repo_id = NAME2REPO[model_name]
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=f"{model_name}.yaml", local_dir=local_dir, **kwargs)
    ckpt_path = hf_hub_download(
        repo_id=repo_id, filename=f"{model_name}.pt", local_dir=local_dir, **kwargs)

    vae_cfg_path = hf_hub_download(
        repo_id="CypressYang/SongBloom", filename="stable_audio_1920_vae.json", local_dir=local_dir, **kwargs)
    vae_ckpt_path = hf_hub_download(
        repo_id="CypressYang/SongBloom", filename="autoencoder_music_dsp1920.ckpt", local_dir=local_dir, **kwargs)

    g2p_path = hf_hub_download(
        repo_id="CypressYang/SongBloom", filename="vocab_g2p.yaml", local_dir=local_dir, **kwargs)

    return {
        "cfg_path": cfg_path,
        "ckpt_path": ckpt_path,
        "vae_cfg_path": vae_cfg_path,
        "vae_ckpt_path": vae_ckpt_path,
        "g2p_path": g2p_path,
    }


def load_config(cfg_file, parent_dir="./") -> DictConfig:
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda x: os.path.splitext(os.path.basename(x))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: OmegaConf.load(x))
    OmegaConf.register_new_resolver("dynamic_path", lambda x: x.replace("???", parent_dir))

    file_cfg = OmegaConf.load(open(cfg_file, 'r')) if cfg_file is not None else OmegaConf.create()
    return file_cfg


def build_train_cfg(args) -> DictConfig:
    train_cfg = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_flow": args.lambda_flow,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps if args.max_steps > 0 else None,
    }
    return OmegaConf.create(train_cfg)


def build_dataset_cfg(args, cfg) -> DatasetConfig:
    return DatasetConfig(
        jsonl_path=args.data_jsonl,
        sample_rate=cfg.sr,
        prompt_len=args.prompt_len,
        max_duration=args.max_duration if args.max_duration > 0 else cfg.max_dur,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        rebuild_cache=args.rebuild_cache,
        segment_strategy=args.segment_strategy,
        clean_lyrics=args.clean_lyrics,
        process_lyrics=args.process_lyrics,
        lyric_processor=args.lyric_processor,
        require_length_match=args.require_length_match,
        log_length_mismatch=args.log_length_mismatch,
        max_mismatch_logs=args.max_mismatch_logs,
        return_length_info=args.verify_lengths,
    )


def maybe_split_jsonl(args) -> tuple[str, str | None]:
    if args.val_jsonl:
        return args.data_jsonl, args.val_jsonl
    if args.val_split <= 0:
        return args.data_jsonl, None

    os.makedirs(args.output_dir, exist_ok=True)
    train_jsonl = os.path.join(args.output_dir, "train_split.jsonl")
    val_jsonl = os.path.join(args.output_dir, "val_split.jsonl")

    if args.overwrite_split or not (os.path.exists(train_jsonl) and os.path.exists(val_jsonl)):
        items = load_jsonl(args.data_jsonl)
        train_items, val_items = split_items(items, args.val_split, args.split_seed)
        write_jsonl(train_jsonl, train_items)
        write_jsonl(val_jsonl, val_items)
    return train_jsonl, val_jsonl


def verify_dataset_lengths(dataset, max_samples: int) -> None:
    mismatches = 0
    total = 0
    max_samples = max_samples if max_samples > 0 else len(dataset)
    for i in range(min(len(dataset), max_samples)):
        sample = dataset[i]
        total += 1
        if sample.get("orig_audio_len") != sample.get("orig_sketch_len"):
            mismatches += 1
    print(f"[verify] checked={total} mismatches={mismatches}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="songbloom_full_150s")
    parser.add_argument("--local-dir", type=str, default="./cache")
    parser.add_argument("--data-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-flow", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--prompt-len", type=float, default=10.0)
    parser.add_argument("--max-duration", type=float, default=0)
    parser.add_argument("--segment-strategy", type=str, default="start", choices=["start", "random"])
    parser.add_argument("--cache-dir", type=str, default="./cache/training")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")

    parser.add_argument("--clean-lyrics", action="store_true")
    parser.add_argument("--process-lyrics", action="store_true")
    parser.add_argument("--lyric-processor", type=str, default="phoneme")

    parser.add_argument("--init-from-pretrained", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--sketch-mode", type=str, default="precomputed", choices=["precomputed", "muq"])
    parser.add_argument("--muq-model-id", type=str, default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--muq-device", type=str, default="cpu")
    parser.add_argument("--muq-sr", type=int, default=24000)
    parser.add_argument("--muq-embed-dim", type=int, default=1024)
    parser.add_argument("--muq-codebook-size", type=int, default=16384)
    parser.add_argument("--muq-vq-path", type=str, default=None)
    parser.add_argument("--muq-vq-decay", type=float, default=0.99)
    parser.add_argument("--muq-commitment-weight", type=float, default=1.0)
    parser.add_argument("--muq-freeze-codebook", action="store_true")
    parser.add_argument("--require-vq-path", action="store_true")

    parser.add_argument("--require-length-match", action="store_true")
    parser.add_argument("--log-length-mismatch", action="store_true")
    parser.add_argument("--max-mismatch-logs", type=int, default=20)
    parser.add_argument("--verify-lengths", action="store_true")
    parser.add_argument("--verify-lengths-max", type=int, default=100)

    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="songbloom")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    downloads = hf_download(args.model_name, args.local_dir)
    cfg = load_config(downloads["cfg_path"], parent_dir=args.local_dir)

    # ensure VAE paths are local
    cfg.vae.vae_cfg = downloads["vae_cfg_path"]
    cfg.vae.vae_ckpt = downloads["vae_ckpt_path"]

    cfg.train = build_train_cfg(args)
    if not hasattr(cfg, "train_dataset"):
        cfg.train_dataset = OmegaConf.create()
    cfg.train_dataset.lyric_processor = args.lyric_processor
    cfg.train_dataset.text_key = getattr(cfg.train_dataset, "text_key", "lyrics")
    cfg.train_dataset.wav_key = getattr(cfg.train_dataset, "wav_key", "prompt_wav")
    cfg.train_dataset.prompt_len = args.prompt_len

    # preprocessing VAE (CPU) for dataset feature extraction
    preprocess_vae = StableVAE(cfg.vae.vae_ckpt, cfg.vae.vae_cfg, sr=cfg.sr)
    preprocess_vae.eval()

    train_jsonl, val_jsonl = maybe_split_jsonl(args)
    args.data_jsonl = train_jsonl
    dataset_cfg = build_dataset_cfg(args, cfg)

    sketch_extractor = None
    if args.sketch_mode == "muq":
        if args.require_vq_path and not args.muq_vq_path:
            raise ValueError("--muq-vq-path is required when --require-vq-path is set.")
        sketch_extractor = ExternalSketchExtractor(
            model_id=args.muq_model_id,
            device=args.muq_device,
            sample_rate=args.muq_sr,
            embedding_dim=args.muq_embed_dim,
            codebook_size=args.muq_codebook_size,
            vq_path=args.muq_vq_path,
            vq_decay=args.muq_vq_decay,
            commitment_weight=args.muq_commitment_weight,
            freeze_codebook=args.muq_freeze_codebook,
        )
    if not args.process_lyrics:
        # lyrics are processed in the LightningModule
        dataset_cfg.process_lyrics = False
    else:
        dataset_cfg.process_lyrics = True

    dataset = SongBloomTrainDataset(
        dataset_cfg,
        vae=preprocess_vae,
        block_size=cfg.model.block_size,
        sketch_extractor=sketch_extractor,
    )
    if args.verify_lengths:
        verify_dataset_lengths(dataset, args.verify_lengths_max)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_training_batch,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = None
    if val_jsonl:
        val_cfg = build_dataset_cfg(args, cfg)
        val_cfg.jsonl_path = val_jsonl
        val_dataset = SongBloomTrainDataset(
            val_cfg,
            vae=preprocess_vae,
            block_size=cfg.model.block_size,
            sketch_extractor=sketch_extractor,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_training_batch,
            pin_memory=args.device.startswith("cuda"),
        )

    model = SongBloom_PL(cfg)
    if args.init_from_pretrained:
        state = torch.load(cfg.pretrained_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded pretrained.")
        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)

    checkpoint_metric = "val/loss" if val_loader is not None else "train/loss"
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        save_top_k=2,
        monitor=checkpoint_metric,
        mode="min",
        save_last=True,
    )

    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
        )

    trainer = pl.Trainer(
        accelerator=args.device,
        devices=1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_cb],
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        logger=logger,
    )

    trainer.fit(model, dataloader, val_dataloaders=val_loader, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
