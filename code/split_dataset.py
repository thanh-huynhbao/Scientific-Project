import numpy as np

from lerobot.datasets import (
    LeRobotDataset,
    split_dataset,
)

def main():
    url = "/home/armi/LRobot/dataset/aic_manipforce"
    dataset = LeRobotDataset(url)

    print('Spilitting dataset into 200ep training set and 20ep test set \n')

    splits = split_dataset(
        dataset,
        splits={'train_set':200/220,'test_set':20/220},
        output_dir='/home/armi/LRobot/dataset'
    )

    print(f"Train split: {splits['train_set'].meta.total_episodes} episodes")
    print(f"Test split: {splits['test_set'].meta.total_episodes} episodes")

if __name__ == '__main__':
    main()