"""Train VQ codebook for MuQ embeddings.

This script:
1. Loads the MuQ model
2. Extracts embeddings from audio files
3. Trains a VectorQuantize layer
4. Saves the trained codebook
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


def load_muq_model(model_id: str, device: str):
    """Load MuQ model."""
    try:
        from muq import MuQ
    except ImportError as exc:
        raise ImportError(
            "MuQ is not installed. Install with: pip install muq"
        ) from exc

    model = MuQ.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return model


def extract_embeddings(muq_model, audio_path: str, device: str, target_sr: int = 24000):
    """Extract MuQ embeddings from audio file."""
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to target sr
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Remove channel dimension for MuQ (expects [batch, samples])
    waveform = waveform.squeeze(0).unsqueeze(0)  # [1, samples]

    # Move to device
    waveform = waveform.to(device)

    # Extract embeddings using forward method
    with torch.no_grad():
        output = muq_model(waveform)
        embeddings = output.last_hidden_state  # [batch, time, dim]

    return embeddings


def create_vq_model(embedding_dim: int, codebook_size: int, decay: float, device: str):
    """Create VectorQuantize model."""
    from vector_quantize_pytorch import VectorQuantize

    vq = VectorQuantize(
        dim=embedding_dim,
        codebook_size=codebook_size,
        decay=decay,
        commitment_weight=1.0,
    )
    vq.to(device)
    return vq


def collect_audio_files(data_dir: str, extensions: list = [".wav", ".mp3", ".flac"]):
    """Collect all audio files from directory."""
    audio_files = []
    for ext in extensions:
        audio_files.extend(Path(data_dir).glob(f"**/*{ext}"))
    return [str(f) for f in audio_files]


def train_vq_codebook(args):
    """Main training function.

    VectorQuantize uses EMA (Exponential Moving Average) for codebook updates,
    so we don't need an optimizer. The codebook is updated automatically
    during forward passes when in training mode.
    """
    device = args.device

    print(f"Loading MuQ model: {args.muq_model_id}")
    muq_model = load_muq_model(args.muq_model_id, device)

    print(f"Creating VQ model with codebook_size={args.codebook_size}")
    vq_model = create_vq_model(
        embedding_dim=args.embedding_dim,
        codebook_size=args.codebook_size,
        decay=args.decay,
        device=device,
    )

    # Collect audio files
    audio_files = collect_audio_files(args.data_dir)
    print(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {args.data_dir}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        total_loss = 0
        total_commit_loss = 0
        num_batches = 0

        pbar = tqdm(audio_files, desc=f"Epoch {epoch+1}/{args.epochs}")

        for audio_path in pbar:
            try:
                # Extract embeddings
                embeddings = extract_embeddings(
                    muq_model, audio_path, device, args.muq_sr
                )

                # embeddings shape: (1, T, D)
                embeddings = embeddings.squeeze(0)  # (T, D)

                # Skip if too short
                if embeddings.shape[0] < args.min_frames:
                    continue

                # Process in chunks to avoid OOM
                chunk_size = args.chunk_size
                for i in range(0, embeddings.shape[0], chunk_size):
                    chunk = embeddings[i:i+chunk_size]
                    if chunk.shape[0] < 2:
                        continue

                    # Forward pass through VQ - EMA updates happen automatically in train mode
                    vq_model.train()
                    # Note: EMA updates happen in forward pass, don't use no_grad
                    quantized, indices, commit_loss = vq_model(chunk.unsqueeze(0))

                    # Reconstruction loss for monitoring (detach to avoid gradients)
                    with torch.no_grad():
                        recon_loss = F.mse_loss(quantized.detach(), chunk.unsqueeze(0))
                        loss = recon_loss

                    total_loss += loss.item()
                    total_commit_loss += commit_loss.item() if isinstance(commit_loss, torch.Tensor) else commit_loss
                    num_batches += 1

                pbar.set_postfix({
                    "loss": total_loss / max(num_batches, 1),
                    "commit": total_commit_loss / max(num_batches, 1),
                })

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(vq_model.state_dict(), args.output)
            print(f"Saved best model to {args.output}")

        # Also save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = args.output.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save(vq_model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print(f"Training complete. Best model saved to {args.output}")

    # Print codebook usage statistics
    vq_model.eval()
    with torch.no_grad():
        print(f"\nCodebook usage statistics:")
        # Sample some audio to check codebook usage
        used_codes = set()
        for audio_path in audio_files[:min(50, len(audio_files))]:
            try:
                embeddings = extract_embeddings(muq_model, audio_path, device, args.muq_sr)
                _, indices, _ = vq_model(embeddings)
                used_codes.update(indices.cpu().numpy().flatten().tolist())
            except:
                continue
        print(f"Unique codes used: {len(used_codes)} / {args.codebook_size}")


def main():
    parser = argparse.ArgumentParser(description="Train VQ codebook for MuQ embeddings")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output", type=str, default="checkpoints/vq_codebook.pt",
                        help="Output path for trained VQ codebook")
    parser.add_argument("--muq-model-id", type=str, default="OpenMuQ/MuQ-large-msd-iter",
                        help="MuQ model ID from HuggingFace")
    parser.add_argument("--muq-sr", type=int, default=24000,
                        help="Sample rate for MuQ input")
    parser.add_argument("--embedding-dim", type=int, default=1024,
                        help="MuQ embedding dimension")
    parser.add_argument("--codebook-size", type=int, default=16384,
                        help="VQ codebook size (must match SongBloom)")
    parser.add_argument("--decay", type=float, default=0.99,
                        help="EMA decay for codebook")
    parser.add_argument("--commitment-weight", type=float, default=1.0,
                        help="Commitment loss weight")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Number of frames per chunk")
    parser.add_argument("--min-frames", type=int, default=50,
                        help="Minimum frames to process")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    args = parser.parse_args()
    train_vq_codebook(args)


if __name__ == "__main__":
    main()
