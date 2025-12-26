import argparse
import os
import torch
import lightning as pl
from omegaconf import OmegaConf, DictConfig
from huggingface_hub import hf_hub_download

from SongBloom.models.songbloom.songbloom_pl import SongBloom_PL
from SongBloom.models.vae_frontend import StableVAE
from SongBloom.training.dataset import DatasetConfig, SongBloomTrainDataset, collate_training_batch

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
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="songbloom_full_150s")
    parser.add_argument("--local-dir", type=str, default="./cache")
    parser.add_argument("--data-jsonl", type=str, required=True)
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
    parser.add_argument("--seed", type=int, default=1234)

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

    dataset_cfg = build_dataset_cfg(args, cfg)
    sketch_extractor = None
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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        save_top_k=2,
        monitor="train/loss",
        mode="min",
        save_last=True,
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
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
