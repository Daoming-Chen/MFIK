import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
import copy

from mfik.model import ConditionalMLP
from mfik.loss import MeanFlowLoss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset .pt")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save checkpoints"
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # MeanFlow params
    parser.add_argument("--time-mu", type=float, default=-0.4)  # Logit-normal mean
    parser.add_argument("--time-sigma", type=float, default=1.0)
    parser.add_argument("--adaptive-p", type=float, default=0.0)

    # Neighborhood Projection params (new from method.md)
    parser.add_argument("--noise-std-min", type=float, default=0.1,
                        help="Minimum noise std for q_ref generation")
    parser.add_argument("--noise-std-max", type=float, default=1.0,
                        help="Maximum noise std for q_ref generation")
    parser.add_argument("--curriculum", action="store_true", default=True,
                        help="Use curriculum learning for noise (large->small)")
    
    # Performance optimization params
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable automatic mixed precision training")
    parser.add_argument("--no-amp", action="store_false", dest="amp",
                        help="Disable automatic mixed precision training")
    parser.add_argument("--compile", action="store_true", default=True,
                        help="Use torch.compile for model optimization")
    parser.add_argument("--no-compile", action="store_false", dest="compile",
                        help="Disable torch.compile")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps")

    return parser.parse_args()


def update_ema(model, ema_model, decay=0.9999):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because it avoids the overhead of single-item
    fetching and collation.
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.tensors[0].device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(t[indices] for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
            
        self.i += self.batch_size
        return batch

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Logger
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # Data
    print(f"Loading data from {args.data} to {args.device}...")
    # Load directly to GPU to avoid CPU-GPU transfer bottleneck
    data = torch.load(args.data, map_location=args.device)
    
    # Use FastTensorDataLoader for efficiency
    train_loader = FastTensorDataLoader(
        data["joints_train"], data["poses_train"],
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = FastTensorDataLoader(
        data["joints_val"], data["poses_val"],
        batch_size=args.batch_size,
        shuffle=False
    )

    n_joints = data["joints_train"].shape[1]
    print(f"Number of joints: {n_joints}")

    # Model
    model = ConditionalMLP(n_joints=n_joints).to(args.device)
    ema_model = copy.deepcopy(model).to(args.device)
    ema_model.requires_grad_(False)

    # Compile model for faster training
    # Use 'reduce-overhead' mode for small models (reduces CPU overhead)
    # 'max-autotune' is better for large models with big matrices
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile (mode=reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Enable TF32 for better performance on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # AMP scaler for mixed precision training
    scaler = GradScaler('cuda', enabled=args.amp)
    use_amp = args.amp and args.device == "cuda"

    # Loss function with Neighborhood Projection
    loss_fn = MeanFlowLoss(
        time_mu=args.time_mu,
        time_sigma=args.time_sigma,
        adaptive_p=args.adaptive_p,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        curriculum_enabled=args.curriculum,
    )

    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        
        # Attempt to load scheduler and scaler states if they exist
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
        # IMPORTANT: When resuming, especially if extending epochs, we must ensure
        # the scheduler knows the correct last_epoch to calculate the LR correctly.
        # If we don't set this, it might reset to initial LR or behave unexpectedly.
        scheduler.last_epoch = start_epoch - 1

    print("Starting training with Neighborhood Projection method...")
    print(f"  Noise std range: [{args.noise_std_min}, {args.noise_std_max}]")
    print(f"  Curriculum learning: {args.curriculum}")
    print(f"  Mixed precision (AMP): {use_amp}")
    print(f"  Batch size: {args.batch_size}")
    
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = torch.tensor(0.0, device=args.device)
        
        # Update curriculum learning epoch
        loss_fn.set_epoch(epoch, args.epochs)

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        for batch_idx, (joints, pose) in enumerate(pbar):
            # Use automatic mixed precision for forward pass
            with autocast('cuda', enabled=use_amp):
                loss, raw_loss = loss_fn(model, joints, pose)
                loss = loss / args.grad_accum_steps
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(model, ema_model)

            train_loss += loss.detach() * args.grad_accum_steps
            global_step += 1

            if global_step % 100 == 0:
                loss_val = loss.item() * args.grad_accum_steps
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/raw_loss", raw_loss.item(), global_step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step
                )
                # Log current noise std for curriculum monitoring
                if args.curriculum:
                    current_noise = loss_fn.get_noise_std(1, joints.device).item()
                    writer.add_scalar("train/noise_std", current_noise, global_step)

                pbar.set_postfix({"loss": f"{loss_val:.4f}", "amp": use_amp})

        avg_train_loss = train_loss.item() / len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = torch.tensor(0.0, device=args.device)
        with torch.no_grad(), autocast('cuda', enabled=use_amp):
            for joints, pose in val_loader:
                # Use EMA model for validation
                loss, _ = loss_fn(ema_model, joints, pose)
                val_loss += loss

        avg_val_loss = val_loss.item() / len(val_loader)
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        print(
            f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f}, Val Loss (EMA): {avg_val_loss:.6f}"
        )

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": vars(args),
                },
                save_path,
            )
            print(f"Saved checkpoint to {save_path}")

    # Save final
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": vars(args),
        },
        final_path,
    )
    print("Training finished!")
    writer.close()


if __name__ == "__main__":
    main()
