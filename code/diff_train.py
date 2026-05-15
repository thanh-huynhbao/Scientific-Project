from pathlib import Path
import torch
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features
import os

os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, saves to file
from collections import defaultdict

class LiveLossPlotter:
    def __init__(self, save_path: Path, update_freq: int = 1):
        self.save_path = save_path
        self.update_freq = update_freq
        self.losses = defaultdict(list)   # {label: [loss, ...]}
        self.steps  = defaultdict(list)   # {label: [step, ...]}

    def update(self, label: str, step: int, loss: float):
        self.losses[label].append(loss)
        self.steps[label].append(step)

        if step % self.update_freq == 0:
            self._plot()

    def _plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        for label in self.losses:
            ax.plot(self.steps[label], self.losses[label], label=label, linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close(fig)
        print(f"Loss plot updated → {self.save_path}")



def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]

def load_checkpoint(checkpoint_path, policy, optimizer):
    ckpt = torch.load(checkpoint_path, map_location="cuda")
    policy.load_state_dict(ckpt["policy_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt["step"]
    print(f"Resumed from step {step}, loss was {ckpt['loss']:.3f}")
    return step

def save_checkpoint(checkpoint_dir, step, policy, optimizer, loss,
                    preprocessor=None, postprocessor=None):
    ckpt_path = checkpoint_dir / f"step_{step:06d}"
    ckpt_path.mkdir(exist_ok=True)

    # 1. Save raw state dicts (lightweight, resumable)
    torch.save({
        "step": step,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, ckpt_path / "checkpoint.pt")

    # 2. Save full pretrained model (inference-ready)
    policy.save_pretrained(ckpt_path)
    if preprocessor is not None:
        preprocessor.save_pretrained(ckpt_path)
    if postprocessor is not None:
        postprocessor.save_pretrained(ckpt_path)

    print(f"Checkpoint saved at step {step} → {ckpt_path}")

def make_policy(input_features, output_features, down_dims, embed_dim, horizon):
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        horizon=horizon,
        n_obs_steps=2,
        n_action_steps=8,
        num_inference_steps=100,
        down_dims=down_dims,
        diffusion_step_embed_dim=embed_dim,
    )
    policy = DiffusionPolicy(cfg)
    return cfg, policy

def train(policy, preprocessor, optimizer, dataloader, ckpt_dir,
          training_steps, log_freq, save_freq, label,
          postprocessor=None, plotter=None):   # ← add postprocessor
    step = 0
    done = False
    it = iter(dataloader)

    print(f"Starting training: {label}")
    while not done:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)

        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"[{label}] step: {step:>6d}  loss: {loss.item():.3f}")

        if plotter is not None:
            plotter.update(label, step, loss.item())

        if step % save_freq == 0 and step > 0:
            save_checkpoint(ckpt_dir, step, policy, optimizer, loss.item(),
                            preprocessor=preprocessor,      # ← pass through
                            postprocessor=postprocessor)    # ← pass through

        step += 1
        if step >= training_steps:
            done = True

    print(f"{label} training complete.")


def main():
    plot_path_50 = Path("/home/armi/LRobot/src/outputs/loss_plot_50.png")
    plotter_50 = LiveLossPlotter(save_path=plot_path_50, update_freq=1)

    plot_path_200 = Path("/home/armi/LRobot/src/outputs/loss_plot_200.png")
    plotter_200 = LiveLossPlotter(save_path=plot_path_200, update_freq=1)

    device = torch.device("cuda")
    dataset_url = '/home/armi/LRobot/dataset/train_set'

    # ── Output dirs ──────────────────────────────────────────────
    out_50  = Path("/home/armi/LRobot/src/outputs/policy_50demos")
    out_200 = Path("/home/armi/LRobot/src/outputs/policy_200demos")
    ckpt_50  = out_50  / "checkpoints"
    ckpt_200 = out_200 / "checkpoints"
    for d in [out_50, out_200, ckpt_50, ckpt_200]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Dataset metadata ─────────────────────────────────────────
    dataset_metadata = LeRobotDatasetMetadata(dataset_url)
    features = dataset_to_policy_features(dataset_metadata.features)
    fps = dataset_metadata.fps

    print(type(features))
    print(len(features))
    for k, i in features.items():
        print(f'{k}\n       {i}')

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features  = {key: ft for key, ft in features.items()
                       if ft.type is not FeatureType.ACTION and key != 'observation.wrench'}

    # ── Build both policies ───────────────────────────────────────
    cfg_50, policy_50  = make_policy(input_features, output_features,
                                down_dims=(256, 512, 1024),
                                embed_dim=128, horizon=96)

    cfg_200, policy_200 = make_policy(input_features, output_features,
                                down_dims=(256, 512, 1024),
                                embed_dim=128, horizon=96)

    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg_50.observation_delta_indices, fps),
        "action":            make_delta_timestamps(cfg_50.action_delta_indices, fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg_50.observation_delta_indices, fps)
        for k in cfg_50.image_features
    }

    # ── Preprocessors ─────────────────────────────────────────────
    preprocessor_50,  postprocessor_50  = make_pre_post_processors(
        cfg_50,  dataset_stats=dataset_metadata.stats, device='cuda')
    preprocessor_200, postprocessor_200 = make_pre_post_processors(
        cfg_200, dataset_stats=dataset_metadata.stats, device='cuda')

    # ── Datasets ──────────────────────────────────────────────────
    dataset_50 = LeRobotDataset(
        dataset_url, delta_timestamps=delta_timestamps,
        episodes=list(range(50)), video_backend="pyav")
    dataset_200 = LeRobotDataset(
        dataset_url, delta_timestamps=delta_timestamps,
        episodes=list(range(200)), video_backend="pyav")

    # ── Dataloaders ───────────────────────────────────────────────
    batch_size = 32
    dataloader_50  = torch.utils.data.DataLoader(
        dataset_50,  batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=4)
    dataloader_200 = torch.utils.data.DataLoader(
        dataset_200, batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=4)

    # ── Optimizers ────────────────────────────────────────────────
    optimizer_50  = cfg_50.get_optimizer_preset().build(policy_50.parameters())
    optimizer_200 = cfg_200.get_optimizer_preset().build(policy_200.parameters())

    # ── Training config ───────────────────────────────────────────
    log_freq  = 1
    # save_freq = 1

    # ── Train Policy 200 second ───────────────────────────────────
    policy_200.train()
    policy_200.to(device)

    train(policy_200, preprocessor_200, optimizer_200, dataloader_200,
        ckpt_200, training_steps=40_000, log_freq=log_freq,
        save_freq=10000, label="Policy 200",
        postprocessor=postprocessor_200, plotter=plotter_200) 
    policy_200.save_pretrained(out_200)
    preprocessor_200.save_pretrained(out_200)
    postprocessor_200.save_pretrained(out_200)
    print(f"Policy 200 saved to {out_200}")

    # Free GPU memory before starting policy 200
    del policy_200, optimizer_200, dataloader_200, dataset_200
    torch.cuda.empty_cache()

    # ── Train Policy 50 first ─────────────────────────────────────
    policy_50.train()
    policy_50.to(device)

    train(policy_50, preprocessor_50, optimizer_50, dataloader_50,
        ckpt_50, training_steps=10_000, log_freq=log_freq,
        save_freq=5000, label="Policy 50",
        postprocessor=postprocessor_50, plotter=plotter_50)    # ← add postprocessor
    policy_50.save_pretrained(out_50)
    preprocessor_50.save_pretrained(out_50)
    postprocessor_50.save_pretrained(out_50)
    print(f"Policy 50 saved to {out_50}")

    print("All done.")

if __name__ == '__main__':
    main()