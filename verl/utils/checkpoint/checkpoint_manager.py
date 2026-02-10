# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import shutil

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.trainer.config import CheckpointConfig
from verl.utils.device import get_device_name, get_torch_device


class BaseCheckpointManager:
    """
    A checkpoint manager that saves and loads the following states in a SPMD way:
    - model
    - optimizer
    - lr_scheduler
    - extra_states

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig | CheckpointConfig = None,
    ):
        self.checkpoint_config = checkpoint_config
        checkpoint_load_contents = checkpoint_config.get("load_contents", None) if checkpoint_config else None
        checkpoint_save_contents = checkpoint_config.get("save_contents", None) if checkpoint_config else None
        if checkpoint_load_contents is None:
            checkpoint_load_contents = ["model", "optimizer", "extra"]
        if checkpoint_save_contents is None:
            checkpoint_save_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = None
        self.previous_saved_paths = []

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_load_contents = checkpoint_load_contents
        self.checkpoint_save_contents = checkpoint_save_contents

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    @property
    def should_save_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_save_contents, indicating the model state should be saved.
        """
        return "model" in self.checkpoint_save_contents

    @property
    def should_save_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_save_contents, indicating the optimizer state should be saved.
        """
        return "optimizer" in self.checkpoint_save_contents

    @property
    def should_save_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_save_contents, indicating the extra state should be saved.
        """
        return "extra" in self.checkpoint_save_contents

    @property
    def should_save_hf_model(self) -> bool:
        """
        Returns True if 'hf_model' is in checkpoint_save_contents, indicating the model should be converted to hf
        model and saved.
        """
        return "hf_model" in self.checkpoint_save_contents

    @property
    def should_load_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_load_contents, indicating the model state should be loaded.
        """
        return "model" in self.checkpoint_load_contents

    @property
    def should_load_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_load_contents, indicating the optimizer state should be loaded.
        """
        return "optimizer" in self.checkpoint_load_contents

    @property
    def should_load_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_load_contents, indicating the extra state should be loaded.
        """
        return "extra" in self.checkpoint_load_contents

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        raise NotImplementedError

    def save_checkpoint(
        self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None
    ):
        raise NotImplementedError

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"
        return local_path is not None, local_path if local_path is not None else hdfs_path

    def remove_previous_save_local_path(self, path):
        if isinstance(path, str):
            path = [path]
        for p in path:
            abs_path = os.path.abspath(p)
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            if not os.path.exists(abs_path):
                continue
            shutil.rmtree(abs_path, ignore_errors=True)
            self.cleanup_empty_global_step_dir(abs_path)

    @staticmethod
    def cleanup_empty_global_step_dir(path: str):
        """Remove the enclosing global_step_* directory if it became empty."""
        parent_dir = os.path.dirname(os.path.abspath(path))
        parent_name = os.path.basename(parent_dir)
        if not parent_name.startswith("global_step_"):
            return
        try:
            if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
        except OSError:
            # Directory may have been removed concurrently or hold new files; ignore safely
            pass

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        if get_device_name() != "cpu":
            rng_state[get_device_name()] = get_torch_device().get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        if get_device_name() != "cpu":
            get_torch_device().set_rng_state(rng_state[get_device_name()])


def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    """
    Return the most recent checkpoint directory based on a tracker file.

    Args:
        path (str): Base directory containing the checkpoint tracker.
        directory_format (str): Template for checkpoint subfolders with one
            placeholder for the iteration number (default "global_step_{}").

    Returns:
        str or None: Full path to the latest checkpoint directory, or
        None if the tracker or checkpoint folder is missing.
    """
    if path is None:
        return None

    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Checkpoint tracker file does not exist: {tracker_file}")
        return None

    with open(tracker_file, "rb") as f:
        iteration = int(f.read().decode())
    ckpt_path = os.path.join(path, directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def get_checkpoint_tracker_filename(root_path: str):
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")


def should_save_ckpt_esi(max_steps_duration: float, save_ckpt_duration: float = 60, redundant_time: float = 0) -> bool:
    """
    Determine if checkpoint should be saved based on capacity esi expiration.

    Args:
        max_steps_duration: Max estimated time (seconds) required to complete one training step
        save_ckpt_duration: Estimated time (seconds) required to save checkpoint (default: 60)
        redundant_time: Additional buffer time (seconds) for unexpected delays (default: 0)
    """
    exp_ts_mlp = os.getenv("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # vemlp
    exp_ts_aws = os.getenv("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # aws
    if exp_ts_mlp:
        try:
            import time

            remaining = float(exp_ts_mlp) - time.time()
        except ValueError:
            return False
        return (
            remaining > 0
            and max_steps_duration > 0
            and remaining <= save_ckpt_duration + max_steps_duration + redundant_time
        )
    elif exp_ts_aws:
        from datetime import datetime, timedelta

        expiration_time = datetime.fromtimestamp(int(exp_ts_aws))
        time_difference = expiration_time - datetime.now()
        threshold_minutes = (save_ckpt_duration + max_steps_duration + redundant_time) / 60
        return time_difference < timedelta(minutes=threshold_minutes)
    else:
        return False


def should_save_ckpt_time_based(
    save_time_interval: float, last_checkpoint_time: float, current_time: float = None
) -> bool:
    """
    Determine if checkpoint should be saved based on elapsed time since last checkpoint.

    Args:
        save_time_interval: Time interval (seconds) between checkpoints. If <= 0, time-based saving is disabled.
        last_checkpoint_time: Timestamp of the last checkpoint (seconds since epoch).
        current_time: Current timestamp (seconds since epoch). If None, uses time.time().

    Returns:
        bool: True if enough time has elapsed since last checkpoint, False otherwise.
    """
    if save_time_interval <= 0:
        return False

    import time

    if current_time is None:
        current_time = time.time()

    elapsed_time = current_time - last_checkpoint_time
    return elapsed_time >= save_time_interval


def extract_step_from_path(checkpoint_path: str) -> int:
    """
    Extract the global step number from a checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint, e.g., "/path/to/global_step_100/actor"
    
    Returns:
        int: The step number, or -1 if not found
    """
    import re
    # Look for "global_step_{number}" in the path
    match = re.search(r'global_step_(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return -1


def determine_preemp_checkpoints_to_delete(
    previous_saved_paths: list, current_step: int, save_freq: int
) -> list:
    """
    Determine which preemptive checkpoints to delete incrementally.
    
    This function implements incremental cleanup where:
    - Checkpoints at multiples of save_freq are NEVER deleted
    - When a new preemptive checkpoint is saved, delete old preemptive checkpoints
      that are before the most recent save_freq multiple
    - Keep the most recent preemptive checkpoint (the one just saved)
    
    Args:
        previous_saved_paths: List of checkpoint paths that have been saved (excluding current)
        current_step: Current step number being saved
        save_freq: Frequency for long-term checkpoint retention (e.g., 20)
    
    Returns:
        list: Paths to delete (only preemptive checkpoints, never save_freq multiples)
    """
    if save_freq <= 0:
        # If save_freq is not set, don't delete anything
        return []
    
    # Extract step numbers from paths
    path_steps = []
    for path in previous_saved_paths:
        step = extract_step_from_path(path)
        if step >= 0:
            path_steps.append((step, path))
    
    if not path_steps:
        return []
    
    # Sort by step number
    path_steps.sort(key=lambda x: x[0])
    
    # Find the most recent save_freq multiple that's <= current_step
    current_save_freq_multiple = (current_step // save_freq) * save_freq
    
    # Find the most recent checkpoint (we'll keep this one)
    most_recent_step, _ = path_steps[-1]
    
    paths_to_delete = []
    
    for step, path in path_steps:
        # NEVER delete checkpoints that are multiples of save_freq
        if step % save_freq == 0:
            continue
        # Keep the most recent checkpoint (even if not a multiple of save_freq)
        if step == most_recent_step:
            continue
        # Delete preemptive checkpoints that are before the current save_freq multiple
        if step < current_save_freq_multiple:
            paths_to_delete.append(path)
    
    return paths_to_delete


def determine_checkpoints_to_keep(
    previous_saved_paths: list, current_step: int, save_freq: int
) -> tuple[list, list]:
    """
    Determine which checkpoints to keep and which to delete based on smart cleanup strategy.
    
    This function implements a strategy where:
    - Checkpoints at multiples of save_freq are kept long-term
    - The most recent checkpoint is always kept (even if not a multiple of save_freq)
    - Intermediate checkpoints between save_freq multiples are deleted when a new
      save_freq multiple is reached
    
    Args:
        previous_saved_paths: List of checkpoint paths that have been saved
        current_step: Current step number being saved
        save_freq: Frequency for long-term checkpoint retention (e.g., 20)
    
    Returns:
        tuple: (paths_to_keep, paths_to_delete)
    """
    if save_freq <= 0:
        # If save_freq is not set, keep all checkpoints (fallback to old behavior)
        return previous_saved_paths, []
    
    # Extract step numbers from paths
    path_steps = []
    for path in previous_saved_paths:
        step = extract_step_from_path(path)
        if step >= 0:
            path_steps.append((step, path))
    
    # Sort by step number
    path_steps.sort(key=lambda x: x[0])
    
    # Determine which checkpoints to keep
    paths_to_keep = []
    paths_to_delete = []
    
    if not path_steps:
        return paths_to_keep, paths_to_delete
    
    # Find the most recent save_freq multiple that's <= current_step
    current_save_freq_multiple = (current_step // save_freq) * save_freq
    
    # Find the most recent checkpoint (for keeping even if not a multiple of save_freq)
    most_recent_step, most_recent_path = path_steps[-1]
    
    for step, path in path_steps:
        # Always keep checkpoints that are multiples of save_freq
        if step % save_freq == 0:
            paths_to_keep.append(path)
        # Keep the most recent checkpoint (even if not a multiple of save_freq)
        elif step == most_recent_step:
            paths_to_keep.append(path)
        # Delete intermediate checkpoints that are before the current save_freq multiple
        elif step < current_save_freq_multiple:
            paths_to_delete.append(path)
        # Keep intermediate checkpoints that are after the current save_freq multiple
        # (they will be deleted when we reach the next save_freq multiple)
        else:
            paths_to_keep.append(path)
    
    return paths_to_keep, paths_to_delete
