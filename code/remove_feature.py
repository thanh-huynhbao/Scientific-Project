import numpy as np

from lerobot.datasets import (
    LeRobotDataset,
    add_features,
    delete_episodes,
    merge_datasets,
    modify_features,
    remove_feature,
    split_dataset,
)


def main():
    dataset_url = "/home/armi/LRobot/dataset/test_set"
    dataset = LeRobotDataset(dataset_url)
    dataset_cleaned = remove_feature(
        dataset, feature_names="observation.wrench", repo_id="/home/armi/LRobot/dataset/testset_cleaned"
    )

    print(f"Features after removal: {list(dataset_cleaned.meta.features.keys())}")
if __name__ == "__main__":
    main()