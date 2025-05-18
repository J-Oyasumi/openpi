"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
from termcolor import cprint
import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
from tqdm import tqdm
import argparse


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    mode: Literal["video", "image"] = "image",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    joints = [
        "torso_lift_joint",
        "head_pan_joint",
        "shoulder_pan_joint",
        "head_tilt_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
        "r_gripper_finger_joint",
        "l_gripper_finger_joint",
    ]
    action_joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
        "r_gripper_finger_joint",
        "head_pan_joint",
        "head_tilt_joint",
        "torso_lift_joint",
        "root_x_axis_joint",
        "root_z_rotation_joint",
    ]
    cameras = [
        "fetch_head",
        "fetch_hand"
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joints),),
            "names": [
                joints,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_joints),),
            "names": [
                action_joints,
            ],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (128, 128, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        }
    
    features["observation.pointcloud"] = {
        "dtype": "float64",
        "shape": (1024, 6),
        "names": [
            "num_points",
            "xyz+rgb",
        ],
    }

    features["observation.extra.obj_pose_wrt_base"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "xyz+quat"
        ]
    }
    features["observation.extra.tcp_pose_wrt_base"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "xyz+quat"
        ]
    }
    features["observation.extra.goal_pos_wrt_base"] = {
        "dtype": "float32",
        "shape": (3,),
        "names": [
            "xyz"
        ]
    }
    features["observation.extra.is_grasped"] = {
        "dtype": "int8",
        "shape": (1,),
        "names": [
            "is_grasped"
        ]
    }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    h5_path: str,
    task: str,
) -> LeRobotDataset:
    with h5py.File(h5_path, 'r') as f:
        cprint(f"Num of trajs: {len(f.keys())}", "green")
        for i in tqdm(range(len(f.keys()))):
            traj = f[f'traj_{i}']
            success = traj['success'][:]
            steps = np.argmax(success) + 1
            assert steps > 1, "Failure Trajectory!"
            del success

            states = traj['obs/agent/qpos'][:steps] 
            actions = traj['actions'][:steps]
            head_imgs = traj['obs/sensor_data/fetch_head/rgb'][:steps]
            hand_imgs = traj['obs/sensor_data/fetch_hand/rgb'][:steps]
            pointclouds = traj['obs/pointcloud'][:steps]
            objs = traj['obs/extra/obj_pose_wrt_base'][:steps]
            tcps = traj['obs/extra/tcp_pose_wrt_base'][:steps]
            goals = traj['obs/extra/goal_pos_wrt_base'][:steps]
            grasped = traj['obs/extra/is_grasped'][:steps][:, None].astype(np.int8)

            for t in range(steps):
                frame = {
                    "task": task,
                    "observation.state": states[t],
                    "action": actions[t],
                    "observation.images.fetch_head": head_imgs[t],
                    "observation.images.fetch_hand": hand_imgs[t],
                    "observation.pointcloud": pointclouds[t],
                    "observation.extra.obj_pose_wrt_base": objs[t],
                    "observation.extra.tcp_pose_wrt_base": tcps[t],
                    "observation.extra.goal_pos_wrt_base": goals[t],
                    "observation.extra.is_grasped": grasped[t],
                }
                dataset.add_frame(frame)

            dataset.save_episode()
            
    return dataset


def port_mshab(
    h5_dir: str,
    repo_id: str,
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    dataset = create_empty_dataset(
        repo_id,
        mode=mode,
        dataset_config=dataset_config,
    )

    task_dir = Path(h5_dir)
    for subtask in task_dir.iterdir():
        if not subtask.is_dir():
            continue
        subtask_dir = subtask.joinpath("train")
        for obj_name in subtask_dir.iterdir():
            if not obj_name.is_dir() or obj_name.name == "all":
                continue

            if subtask.name == "close" and obj_name.name == "fridge":
                continue
            
            task_name = subtask.name + "_" + obj_name.name
            cprint(f"Processing {task_name}", "green")
            h5_path = list(obj_name.glob("*.h5"))[0]
            
            dataset = populate_dataset(
                dataset,
                h5_path=h5_path,
                task=task_name,
            )
            
    dataset.consolidate(run_compute_stats=False)
        
    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "h5_dir",
        type=str,
        help="Path to the raw data directory",
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="The repo id to push the dataset to",
    )
    args = parser.parse_args()
    
    if not Path(args.h5_dir).exists():
        raise ValueError("h5 directory does not exist")
    
    port_mshab(
        h5_dir=args.h5_dir,
        repo_id=args.repo_id,
    )
