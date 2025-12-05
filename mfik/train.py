import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

    return parser.parse_args()


def update_ema(model, ema_model, decay=0.9999):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Logger
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # Data
    print(f"Loading data from {args.data} to {args.device}...")
    # Load directly to GPU to avoid CPU-GPU transfer bottleneck
    data = torch.load(args.data, map_location=args.device)
    
    train_dataset = TensorDataset(data["joints_train"], data["poses_train"])
    val_dataset = TensorDataset(data["joints_val"], data["poses_val"])

    # Use num_workers=0 because data is already on GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    n_joints = data["joints_train"].shape[1]
    print(f"Number of joints: {n_joints}")

    # Model
    model = ConditionalMLP(n_joints=n_joints).to(args.device)
    ema_model = copy.deepcopy(model).to(args.device)
    ema_model.requires_grad_(False)

    # Compile model for faster training
    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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

    print("Starting training with Neighborhood Projection method...")
    print(f"  Noise std range: [{args.noise_std_min}, {args.noise_std_max}]")
    print(f"  Curriculum learning: {args.curriculum}")
    
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = torch.tensor(0.0, device=args.device)
        
        # Update curriculum learning epoch
        loss_fn.set_epoch(epoch, args.epochs)

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for joints, pose in pbar:
            # Data is already on device
            # joints = joints.to(args.device, dtype=torch.float32)
            # pose = pose.to(args.device, dtype=torch.float32)

            loss, raw_loss = loss_fn(model, joints, pose)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            update_ema(model, ema_model)

            train_loss += loss.detach()
            global_step += 1

            if global_step % 100 == 0:
                loss_val = loss.item()
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/raw_loss", raw_loss.item(), global_step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step
                )
                # Log current noise std for curriculum monitoring
                if args.curriculum:
                    current_noise = loss_fn.get_noise_std(1, joints.device).item()
                    writer.add_scalar("train/noise_std", current_noise, global_step)

                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_train_loss = train_loss.item() / len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = torch.tensor(0.0, device=args.device)
        with torch.no_grad():
            for joints, pose in val_loader:
                # Data is already on device
                # joints = joints.to(args.device, dtype=torch.float32)
                # pose = pose.to(args.device, dtype=torch.float32)
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
            "config": vars(args),
        },
        final_path,
    )
    print("Training finished!")
    writer.close()


if __name__ == "__main__":
    main()
