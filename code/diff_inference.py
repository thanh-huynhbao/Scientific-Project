import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.configs import FeatureType

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


MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20

def space():
    print("----------------------------")

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]

def main():
    device = torch.device("cuda")
    model_url = "/home/armi/LRobot/src/outputs/policy_200demos"

    model = DiffusionPolicy.from_pretrained(model_url)

    dataset_url = "/home/armi/LRobot/dataset/train_set"
    dataset_metadata = LeRobotDatasetMetadata(dataset_url)
    preprocess, postprocess = make_pre_post_processors(
        model.config, model_url, dataset_stats=dataset_metadata.stats
    )

    # Extract inference input from test set
    test_url = "/home/armi/LRobot/dataset/test_set"
    test_metadata = LeRobotDatasetMetadata(test_url)

    # total eps
    total_eps = test_metadata.total_episodes
    print(total_eps)
    space()
    
    testset = LeRobotDataset(test_url, video_backend="pyav")

    episode_index = 0
    from_idx = testset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = testset.meta.episodes["dataset_to_index"][episode_index]
    print(f"from idx: {from_idx}; to: {to_idx}")
    space()

    camera_key = testset.meta.camera_keys
    print(camera_key)
    space()

    features = dataset_to_policy_features(test_metadata.features)
    print(f"Total Features:\n\n{features}\n\nLength of Feature {len(features)}")
    space()
    obs = []
    for k, i in features.items():
        print(f"\n{k}")
        obs.append(testset[1][k])

    space()
    obs = obs[:4]
    print(obs)

    
    


if __name__ == "__main__":
    main()