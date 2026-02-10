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

import glob
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
import warnings
from dataclasses import asdict, dataclass
from multiprocessing import Process
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )
        
        # Initialize async checkpointing state
        self.async_save_mode = (
            checkpoint_config.get("async_save_mode", None) if checkpoint_config else None
        )
        self.async_cleanup = (
            checkpoint_config.get("async_cleanup", False) if checkpoint_config else False
        )
        self.thread_debug = (
            checkpoint_config.get("thread_debug", True) if checkpoint_config else True
        )
        
        # Used for async saving
        self._shm_save_hash = hex(hash(str(time.time() + self.rank)))[-8:]
        self._save_process = None
        self._cleanup_process = None

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # check if the checkpoint_load_contents is valid
        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        # every rank download its own checkpoint
        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_model
            else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if self.should_load_model:
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                self.model.load_state_dict(model_state_dict)
                log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)

            if self.should_load_optimizer:
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state_dict)
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

        if self.should_load_extra:
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        if self.rank == 0 and del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    def __getstate__(self):
        """Capture what is normally pickled, removing unpicklable/problematic variables."""
        state = self.__dict__.copy()
        # Remove unpicklable/problematic variables
        state['_save_process'] = None
        state['_cleanup_process'] = None
        return state

    @staticmethod
    def _async_shm_save(state_dict_path: str, rank: int, save_dir: str, thread_debug: bool = True):
        """Process target for async state dict saving.
        
        Note: This is a static method because it's used as a Process target.
        Instance variables are not available in the child process.
        """
        import time
        start_time = time.time()
        if thread_debug:
            print(f"Rank {rank} [ASYNC_SAVE_START] saving state dict from {state_dict_path} to {save_dir}")
        
        # Copy checkpoint files from temp location to final save dir
        copy_start = time.time()
        files_to_copy = list(Path(state_dict_path).glob("*"))
        if logger:
            logger.info(f"Rank {rank} [ASYNC_SAVE] Found {len(files_to_copy)} files to copy")
        
        for file in files_to_copy:
            file_start = time.time()
            shutil.move(str(file), str(Path(save_dir) / file.name))
            file_time = time.time() - file_start
            if file_time > 1.0 and logger:  # Log slow file copies
                logger.info(f"Rank {rank} [ASYNC_SAVE] Slow file copy: {file.name} took {file_time:.2f}s")
        
        copy_time = time.time() - copy_start
        if logger:
            logger.info(f"Rank {rank} [ASYNC_SAVE] File copy completed in {copy_time:.2f}s")
        
        # Clean up temp directory
        cleanup_start = time.time()
        shutil.rmtree(state_dict_path)
        cleanup_time = time.time() - cleanup_start
        if logger:
            logger.info(f"Rank {rank} [ASYNC_SAVE] Temp dir cleanup took {cleanup_time:.2f}s")
        
        # Touch save_dir / ckpt_{rank}.complete to mark done
        (Path(save_dir) / f"ckpt_{rank}.complete").touch()
        
        total_time = time.time() - start_time
        if logger:
            logger.info(f"Rank {rank} [ASYNC_SAVE_COMPLETE] Total async save time: {total_time:.2f}s")
        if thread_debug:
            print(f"Rank {rank} saved state dict to {save_dir} in {total_time:.2f}s")

    def _dump_sharded_checkpoint(self, target_dir: str):
        """Write sharded model/optimizer/extra state dicts to ``target_dir``."""
        import time
        dump_start = time.time()
        target_dir = local_mkdir_safe(target_dir)
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(target_dir, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(target_dir, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(target_dir, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:
                    model_start = time.time()
                    model_state_dict = self.model.state_dict()
                    model_dict_time = time.time() - model_start
                    logger.info(f"Rank {self.rank} [CHECKPOINT_DUMP] Model state_dict() took {model_dict_time:.2f}s")
                    
                    save_start = time.time()
                    torch.save(model_state_dict, model_path)
                    save_time = time.time() - save_start
                    logger.info(f"Rank {self.rank} [CHECKPOINT_DUMP] Model torch.save() took {save_time:.2f}s, size: {os.path.getsize(model_path) / 1024**2:.2f} MB")
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:
                    optim_start = time.time()
                    optimizer_state_dict = self.optimizer.state_dict()
                    optim_dict_time = time.time() - optim_start
                    logger.info(f"Rank {self.rank} [CHECKPOINT_DUMP] Optimizer state_dict() took {optim_dict_time:.2f}s")
                    
                    save_start = time.time()
                    torch.save(optimizer_state_dict, optim_path)
                    save_time = time.time() - save_start
                    logger.info(f"Rank {self.rank} [CHECKPOINT_DUMP] Optimizer torch.save() took {save_time:.2f}s, size: {os.path.getsize(optim_path) / 1024**2:.2f} MB")
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    extra_start = time.time()
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    extra_dict_time = time.time() - extra_start
                    logger.info(f"Rank {self.rank} [CHECKPOINT_DUMP] Extra state_dict() took {extra_dict_time:.2f}s")
                    
                    save_start = time.time()
                    torch.save(extra_state_dict, extra_path)
                    save_time = time.time() - save_start
                    logger.info(f"Rank {self.rank} [CHECKPOINT_DUMP] Extra torch.save() took {save_time:.2f}s")
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)
        
        dump_time = time.time() - dump_start
        logger.info(f"Rank {self.rank} [CHECKPOINT_DUMP_COMPLETE] Total dump time: {dump_time:.2f}s")

    def remove_folders(self, folders_to_remove: Tuple[str], rank: int):
        """Process target for folder removal."""
        if self.thread_debug:
            print(f"Rank {rank} starting cleanup of {len(folders_to_remove)} folders")
        for folder_str in folders_to_remove:
            folder = Path(folder_str)
            if not folder.exists():
                continue
            for file in folder.iterdir():
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
            folder.rmdir()
            self.cleanup_empty_global_step_dir(str(folder))
        if self.thread_debug:
            print(f"Rank {rank} completed cleanup")

    @staticmethod
    def extract_step_number(directory_name: str) -> Optional[int]:
        """Extract step number from directory name matching 'global_step_<number>' pattern."""
        match = re.match(r'global_step_(\d+)$', directory_name)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _static_find_checkpoint_directories(root_dir: str) -> List[Tuple[str, int]]:
        """Static version of find_checkpoint_directories for use in async processes."""
        checkpoints = []
        root_path = Path(root_dir)
        
        if not root_path.exists():
            if logger:
                logger.warning(f"[CLEANUP_ASYNC] Root directory does not exist: {root_dir}")
            return checkpoints
        
        # Look for global_step_* directories
        # Note: We need to look inside subdirectories like "actor", "critic" etc.
        # The structure is: root_dir/global_step_X/actor/, root_dir/global_step_X/critic/
        # So we look for global_step_* directories directly in root_dir
        try:
            items = list(root_path.iterdir())
            if logger:
                logger.info(f"[CLEANUP_ASYNC] Found {len(items)} items in {root_dir}")
            
            for item in items:
                if item.is_dir():
                    step_number = FSDPCheckpointManager.extract_step_number(item.name)
                    if step_number is not None:
                        checkpoints.append((str(item), step_number))
                        if logger:
                            logger.debug(f"[CLEANUP_ASYNC] Found checkpoint directory: {item.name} (step {step_number})")
        except Exception as e:
            if logger:
                logger.error(f"[CLEANUP_ASYNC] Error scanning directory {root_dir}: {e}")
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x[1])
        if logger:
            logger.info(f"[CLEANUP_ASYNC] Found {len(checkpoints)} checkpoint directories total")
        return checkpoints

    @staticmethod
    def _static_delete_optim_files_from_checkpoint(checkpoint_dir: str, rank: int) -> int:
        """Static version of delete_optim_files_from_checkpoint for use in async processes.
        
        Only rank 0 deletes files. Deletes all optim_*.pt files matching the pattern.
        
        Args:
            checkpoint_dir: Path to checkpoint directory (e.g., global_step_100/actor)
            rank: Rank number (only rank 0 performs deletion)
            
        Returns:
            Number of optimizer files deleted.
        """
        # Only rank 0 deletes files to avoid synchronization
        if rank != 0:
            return 0
        
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            return 0
        
        # Delete all optimizer files matching optim_*.pt pattern
        deleted_count = 0
        try:
            optim_files = list(checkpoint_path.glob("optim_*.pt"))
            for optim_file in optim_files:
                try:
                    optim_file.unlink()
                    if logger:
                        logger.info(f"Rank {rank} [CLEANUP_ASYNC] Deleted optimizer file: {optim_file.name}")
                    deleted_count += 1
                except (OSError, PermissionError) as e:
                    if logger:
                        logger.warning(f"Rank {rank} [CLEANUP_ASYNC] Error deleting {optim_file.name}: {e}")
            
            if deleted_count > 0 and logger:
                logger.info(f"Rank {rank} [CLEANUP_ASYNC] Deleted {deleted_count} optimizer files from {checkpoint_dir}")
        except Exception as e:
            if logger:
                logger.warning(f"Rank {rank} [CLEANUP_ASYNC] Error listing optimizer files in {checkpoint_dir}: {e}")
        
        return deleted_count

    @staticmethod
    def _async_two_pass_cleanup_impl(root_dir: str, current_step: int, save_freq: int, patience: int, world_size: int, rank: int):
        """Static implementation of two-pass cleanup for use in async processes.
        
        Only rank 0 performs cleanup operations to avoid synchronization overhead.
        """
        import time
        cleanup_start = time.time()
        
        # Only rank 0 performs cleanup
        if rank != 0:
            return
        
        # Calculate threshold
        threshold = current_step - patience
        
        if threshold <= 0:
            # Not enough steps to clean up yet
            return
        
        # Find all checkpoint directories (ONCE - reused for both passes)
        scan_start = time.time()
        if logger:
            logger.info(f"Rank {rank} [CLEANUP_ASYNC] Scanning root_dir: {root_dir}")
            logger.info(f"Rank {rank} [CLEANUP_ASYNC] root_dir exists: {Path(root_dir).exists()}")
        
        checkpoints = FSDPCheckpointManager._static_find_checkpoint_directories(root_dir)
        scan_time = time.time() - scan_start
        if logger:
            logger.info(f"Rank {rank} [CLEANUP_ASYNC] Directory scan found {len(checkpoints)} checkpoints in {scan_time:.2f}s")
            if checkpoints:
                logger.info(f"Rank {rank} [CLEANUP_ASYNC] Checkpoint steps found: {[step for _, step in checkpoints]}")
        
        if not checkpoints:
            if logger:
                logger.warning(f"Rank {rank} [CLEANUP_ASYNC] No checkpoints found in {root_dir}, skipping cleanup")
            return
        
        if logger:
            logger.info(
                f"Rank {rank} [CLEANUP_ASYNC] Two-pass cleanup: current_step={current_step}, threshold={threshold}, "
                f"save_freq={save_freq}, patience={patience}"
            )
        
        # PASS 1: Delete entire directories
        pass1_start = time.time()
        dirs_to_delete = []
        for ckpt_path, step_number in checkpoints:
            # Delete if: step < threshold AND step % save_freq != 0
            if step_number < threshold and step_number % save_freq != 0:
                dirs_to_delete.append(ckpt_path)
        
        if dirs_to_delete:
            if logger:
                logger.info(f"Rank {rank} [CLEANUP_ASYNC] Pass 1: Deleting {len(dirs_to_delete)} entire checkpoint directories")
            for ckpt_path in dirs_to_delete:
                try:
                    delete_start = time.time()
                    shutil.rmtree(ckpt_path)
                    delete_time = time.time() - delete_start
                    if logger:
                        logger.info(f"Rank {rank} [CLEANUP_ASYNC] Deleted checkpoint directory: {ckpt_path} in {delete_time:.2f}s")
                except (OSError, PermissionError) as e:
                    if logger:
                        logger.warning(f"Rank {rank} [CLEANUP_ASYNC] Error deleting {ckpt_path}: {e}")
        
        pass1_time = time.time() - pass1_start
        if logger:
            logger.info(f"Rank {rank} [CLEANUP_ASYNC] Pass 1 completed in {pass1_time:.2f}s")
        
        # PASS 2: Delete optimizer files from remaining directories below threshold
        # Reuse checkpoints list from Pass 1, filtering out deleted directories
        pass2_start = time.time()
        optim_files_deleted = 0
        dirs_processed = 0
        
        for ckpt_path, step_number in checkpoints:
            # Process directories where step < threshold AND directory still exists (survived pass 1)
            if step_number < threshold and Path(ckpt_path).exists():
                deleted = FSDPCheckpointManager._static_delete_optim_files_from_checkpoint(
                    ckpt_path, rank
                )
                if deleted > 0:
                    optim_files_deleted += deleted
                    dirs_processed += 1
        
        pass2_time = time.time() - pass2_start
        if dirs_processed > 0 and logger:
            logger.info(
                f"Rank {rank} [CLEANUP_ASYNC] Pass 2: Deleted {optim_files_deleted} optimizer files from {dirs_processed} checkpoint directories in {pass2_time:.2f}s"
            )
        
        cleanup_total_time = time.time() - cleanup_start
        if logger:
            logger.info(f"Rank {rank} [CLEANUP_ASYNC_COMPLETE] Total cleanup time: {cleanup_total_time:.2f}s")

    def find_checkpoint_directories(self, root_dir: str) -> List[Tuple[str, int]]:
        """Find all checkpoint directories with their step numbers.
        
        Returns:
            List of tuples (directory_path, step_number) sorted by step number.
        """
        checkpoints = []
        root_path = Path(root_dir)
        
        if not root_path.exists():
            return checkpoints
        
        # Look for global_step_* directories
        for item in root_path.iterdir():
            if item.is_dir():
                step_number = self.extract_step_number(item.name)
                if step_number is not None:
                    checkpoints.append((str(item), step_number))
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x[1])
        return checkpoints

    def delete_optim_files_from_checkpoint(self, checkpoint_dir: str) -> int:
        """Delete all optim_*.pt files from a checkpoint directory.
        
        Only rank 0 performs deletion to avoid synchronization overhead.
        Rank 0 deletes all optimizer files matching optim_*.pt pattern.
        
        Args:
            checkpoint_dir: Path to checkpoint directory (e.g., global_step_100/actor)
            
        Returns:
            Number of optimizer files deleted.
        """
        # Only rank 0 deletes files to avoid synchronization
        if self.rank != 0:
            return 0
        
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            return 0
        
        # Delete all optimizer files matching optim_*.pt pattern
        deleted_count = 0
        try:
            optim_files = list(checkpoint_path.glob("optim_*.pt"))
            for optim_file in optim_files:
                try:
                    optim_file.unlink()
                    logger.info(f"Rank {self.rank} [CLEANUP] Deleted optimizer file: {optim_file.name}")
                    deleted_count += 1
                except (OSError, PermissionError) as e:
                    logger.warning(f"Rank {self.rank} [CLEANUP] Error deleting {optim_file.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Rank {self.rank} [CLEANUP] Deleted {deleted_count} optimizer files from {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Rank {self.rank} [CLEANUP] Error listing optimizer files in {checkpoint_dir}: {e}")
        
        return deleted_count

    def two_pass_cleanup(self, root_dir: str, current_step: int, save_freq: int, patience: int = 4):
        """Perform two-pass cleanup of old checkpoints.
        
        Pass 1: Delete entire checkpoint directories where step < threshold AND step % save_freq != 0
        Pass 2: Delete optimizer files from remaining directories where step < threshold
        
        Args:
            root_dir: Root directory containing checkpoint subdirectories
            current_step: Current global step number
            save_freq: Checkpoint save frequency (interval)
            patience: Number of recent steps to keep from current step
        """
        import time
        cleanup_start = time.time()
        
        if self.rank != 0:
            # Only rank 0 performs cleanup
            return
        
        # Calculate threshold
        threshold = current_step - patience
        
        if threshold <= 0:
            # Not enough steps to clean up yet
            return
        
        # Find all checkpoint directories (ONCE - reused for both passes)
        scan_start = time.time()
        logger.info(f"Rank {self.rank} [CLEANUP_SYNC] Scanning root_dir: {root_dir}")
        logger.info(f"Rank {self.rank} [CLEANUP_SYNC] root_dir exists: {Path(root_dir).exists()}")
        
        checkpoints = self.find_checkpoint_directories(root_dir)
        scan_time = time.time() - scan_start
        logger.info(f"Rank {self.rank} [CLEANUP_SYNC] Directory scan found {len(checkpoints)} checkpoints in {scan_time:.2f}s")
        if checkpoints:
            logger.info(f"Rank {self.rank} [CLEANUP_SYNC] Checkpoint steps found: {[step for _, step in checkpoints]}")
        
        if not checkpoints:
            logger.warning(f"Rank {self.rank} [CLEANUP_SYNC] No checkpoints found in {root_dir}, skipping cleanup")
            return
        
        log_with_rank(
            f"Two-pass cleanup: current_step={current_step}, threshold={threshold}, "
            f"save_freq={save_freq}, patience={patience}",
            rank=self.rank,
            logger=logger
        )
        
        # PASS 1: Delete entire directories
        pass1_start = time.time()
        dirs_to_delete = []
        for ckpt_path, step_number in checkpoints:
            # Delete if: step < threshold AND step % save_freq != 0
            if step_number < threshold and step_number % save_freq != 0:
                dirs_to_delete.append(ckpt_path)
        
        if dirs_to_delete:
            log_with_rank(
                f"Pass 1: Deleting {len(dirs_to_delete)} entire checkpoint directories",
                rank=self.rank,
                logger=logger
            )
            for ckpt_path in dirs_to_delete:
                try:
                    delete_start = time.time()
                    shutil.rmtree(ckpt_path)
                    delete_time = time.time() - delete_start
                    log_with_rank(
                        f"Deleted checkpoint directory: {ckpt_path} in {delete_time:.2f}s",
                        rank=self.rank,
                        logger=logger
                    )
                    # Remove from previous_saved_paths if present
                    if ckpt_path in self.previous_saved_paths:
                        self.previous_saved_paths.remove(ckpt_path)
                except (OSError, PermissionError) as e:
                    log_with_rank(
                        f"Error deleting {ckpt_path}: {e}",
                        rank=self.rank,
                        logger=logger
                    )
        
        pass1_time = time.time() - pass1_start
        logger.info(f"Rank {self.rank} [CLEANUP_SYNC] Pass 1 completed in {pass1_time:.2f}s")
        
        # PASS 2: Delete optimizer files from remaining directories below threshold
        # Reuse checkpoints list from Pass 1, filtering out deleted directories
        # Only rank 0 performs deletion to avoid synchronization
        pass2_start = time.time()
        optim_files_deleted = 0
        dirs_processed = 0
        
        if self.rank == 0:
            for ckpt_path, step_number in checkpoints:
                # Process directories where step < threshold AND directory still exists (survived pass 1)
                if step_number < threshold and Path(ckpt_path).exists():
                    deleted = self.delete_optim_files_from_checkpoint(ckpt_path)
                    if deleted > 0:
                        optim_files_deleted += deleted
                        dirs_processed += 1
        
        pass2_time = time.time() - pass2_start
        if dirs_processed > 0:
            log_with_rank(
                f"Pass 2: Deleted {optim_files_deleted} optimizer files from {dirs_processed} checkpoint directories in {pass2_time:.2f}s",
                rank=self.rank,
                logger=logger
            )
        
        cleanup_total_time = time.time() - cleanup_start
        logger.info(f"Rank {self.rank} [CLEANUP_SYNC_COMPLETE] Total cleanup time: {cleanup_total_time:.2f}s")

    def wipe_shm(self):
        """Clean up shared memory checkpoint files."""
        if self.async_save_mode == "shm":
            if self._save_process is not None:
                self._save_process.terminate()
            shm_base = '/dev/shm'
            if os.path.exists(shm_base):
                for item in os.listdir(shm_base):
                    if item.startswith(self._shm_save_hash):
                        full_path = os.path.join(shm_base, item)
                        if os.path.isdir(full_path):
                            shutil.rmtree(full_path)
                        else:
                            os.remove(full_path)

    def __del__(self):
        """Cleanup on deletion."""
        if self._save_process is not None:
            self._save_process.join()
        # Clean up shm
        if self.async_save_mode == "shm":
            self.wipe_shm()
        # Update destructor to also wait for cleanup
        if self._cleanup_process is not None:
            self._cleanup_process.join()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None, force_sync_save: bool = False, save_freq: int = -1, patience: int = 4, is_preemp_checkpoint: bool = False):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep` (if save_freq is not set).
        When save_freq is set, performs two-pass cleanup at checkpoint intervals to save disk space.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain (used when save_freq <= 0).
            force_sync_save: Force synchronous save even if async_save_mode is enabled.
            save_freq: Checkpoint save frequency. When > 0, enables two-pass cleanup strategy.
            patience: Number of recent steps to keep during cleanup (default: 4).
            is_preemp_checkpoint: Unused, kept for backward compatibility.
        """
        if local_path is None:
            return

        import time
        save_checkpoint_start = time.time()
        logger.info(f"Rank {self.rank} [SAVE_CHECKPOINT_START] global_step={global_step}, local_path={local_path}")

        # Wait for any previous save to complete if async
        # Similar to reference code: check if async save is in progress and wait for it
        if self.async_save_mode is not None:
            if self._save_process is not None:
                wait_start = time.time()
                # Check if process is still running (similar to checking future.done() in reference)
                if self._save_process.is_alive():
                    logger.info(f"Rank {self.rank} [SAVE_CHECKPOINT] Waiting for previous async save to complete...")
                    self._save_process.join()  # Blocks until process completes (similar to future.result())
                else:
                    # Process already completed, just clean up
                    self._save_process.join()  # Still call join() to clean up process resources
                    logger.info(f"Rank {self.rank} [SAVE_CHECKPOINT] Previous async save already completed")
                
                wait_time = time.time() - wait_start
                logger.info(f"Rank {self.rank} [SAVE_CHECKPOINT] Waited {wait_time:.2f}s for previous async save")
                if wait_time > 5.0:
                    logger.warning(f"Rank {self.rank} [SAVE_CHECKPOINT] Long wait time for async save: {wait_time:.2f}s")
                self._save_process = None
        
        barrier_start = time.time()
        torch.distributed.barrier()  # Ensure all ranks are ready to start new save
        barrier_time = time.time() - barrier_start
        if barrier_time > 1.0:
            logger.info(f"Rank {self.rank} [SAVE_CHECKPOINT] Barrier wait time: {barrier_time:.2f}s")

        # record the previous global step
        self.previous_global_step = global_step

        # max_ckpt_to_keep cleanup (only when save_freq is not set)
        if save_freq <= 0 and (
            max_ckpt_to_keep
            and isinstance(max_ckpt_to_keep, int)
            and max_ckpt_to_keep > 0
            and len(self.previous_saved_paths) >= max_ckpt_to_keep
        ):
            if self.rank == 0:
                keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
                folders_to_remove = self.previous_saved_paths[:keep_start]
                
                if self.async_cleanup:
                    # Wait for any previous cleanup to complete
                    if self._cleanup_process is not None:
                        logger.info(f"Rank {self.rank} waiting for previous cleanup to complete...")
                        self._cleanup_process.join()
                        self._cleanup_process = None
                    
                    # Launch async cleanup
                    str_folders_to_remove = tuple([str(folder) for folder in folders_to_remove])
                    self._cleanup_process = Process(
                        target=self.remove_folders,
                        args=(str_folders_to_remove, self.rank)
                    )
                    self._cleanup_process.start()
                else:
                    self.remove_previous_save_local_path(folders_to_remove)
                
                self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        async_save_started = False
        if self.async_save_mode == "shm" and (not force_sync_save):
            try:
                async_setup_start = time.time()
                tmp_dir = Path("/dev/shm") / f"{self._shm_save_hash}_rank{self.rank}_step{global_step}_{uuid.uuid4().hex}"
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Rank {self.rank} [ASYNC_SAVE_SETUP] Created tmp dir {tmp_dir} in {time.time() - async_setup_start:.2f}s")

                # Dump checkpoint shards to shared memory
                dump_start = time.time()
                self._dump_sharded_checkpoint(str(tmp_dir))
                dump_time = time.time() - dump_start
                logger.info(f"Rank {self.rank} [ASYNC_SAVE_SETUP] Dump to shm completed in {dump_time:.2f}s")

                # Launch async copy process
                # Pass thread_debug as argument since _async_shm_save is static
                process_start = time.time()
                self._save_process = Process(
                    target=self._async_shm_save,
                    args=(str(tmp_dir), self.rank, local_path, self.thread_debug)
                )
                self._save_process.start()
                process_time = time.time() - process_start
                logger.info(f"Rank {self.rank} [ASYNC_SAVE_SETUP] Process started in {process_time:.2f}s, PID: {self._save_process.pid}")

                async_save_started = True
            except Exception as e:
                log_with_rank(
                    f"Async checkpointing failed: {e}, falling back to synchronous save",
                    rank=self.rank,
                    logger=logger
                )
                force_sync_save = True

        if not async_save_started:
            sync_dump_start = time.time()
            self._dump_sharded_checkpoint(local_path)
            sync_dump_time = time.time() - sync_dump_start
            logger.info(f"Rank {self.rank} [SYNC_SAVE] Synchronous dump completed in {sync_dump_time:.2f}s")

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

            model_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
            # loaded from the Hub.
            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)

        # wait for everyone to dump to local
        barrier_start = time.time()
        torch.distributed.barrier()
        barrier_time = time.time() - barrier_start
        if barrier_time > 1.0:
            logger.info(f"Rank {self.rank} [SAVE_CHECKPOINT] Post-dump barrier wait time: {barrier_time:.2f}s")

        if self.should_save_hf_model:
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    # Handle different transformers versions for Vision2Seq models
                    import transformers
                    from packaging import version

                    if version.parse(transformers.__version__) >= version.parse("4.54.0"):
                        # transformers >= 4.54.0 uses AutoModelForImageTextToText
                        from transformers import AutoModelForImageTextToText

                        auto_model_cls = AutoModelForImageTextToText
                    else:
                        # transformers < 4.54.0 uses AutoModelForVision2Seq
                        from transformers import AutoModelForVision2Seq

                        auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            barrier_start = time.time()
            torch.distributed.barrier()
            barrier_time = time.time() - barrier_start
            if barrier_time > 1.0:
                logger.info(f"Rank {self.rank} [SAVE_CHECKPOINT] Post-HF-model barrier wait time: {barrier_time:.2f}s")

        self.previous_saved_paths.append(local_path)
        
        # Two-pass cleanup strategy (when save_freq is set)
        # Triggered when current step is a checkpoint interval
        if save_freq > 0 and global_step % save_freq == 0:
            cleanup_start = time.time()
            # Get the parent directory containing all global_step_* subdirectories
            # local_path is like: /checkpoints/global_step_76/actor
            # We need: /checkpoints (parent.parent) to find all global_step_* directories
            parent_dir = str(Path(local_path).parent.parent)
            
            # Debug logging
            logger.info(f"Rank {self.rank} [CLEANUP] local_path={local_path}, parent_dir={parent_dir}")
            logger.info(f"Rank {self.rank} [CLEANUP] parent_dir exists: {Path(parent_dir).exists()}")
            
            # Perform two-pass cleanup (async if enabled, sync otherwise)
            if self.rank == 0:
                log_with_rank(
                    f"Triggering two-pass cleanup at step {global_step} (save_freq={save_freq}, patience={patience}, async={self.async_cleanup})",
                    rank=self.rank,
                    logger=logger
                )
                
                if self.async_cleanup:
                    # Wait for any previous cleanup to complete
                    if self._cleanup_process is not None:
                        cleanup_wait_start = time.time()
                        logger.info(f"Rank {self.rank} [CLEANUP] Waiting for previous cleanup to complete...")
                        self._cleanup_process.join()
                        cleanup_wait_time = time.time() - cleanup_wait_start
                        logger.info(f"Rank {self.rank} [CLEANUP] Waited {cleanup_wait_time:.2f}s for previous cleanup")
                        self._cleanup_process = None
                    
                    # Launch async cleanup process
                    # Use static implementation that can be pickled
                    process_start = time.time()
                    self._cleanup_process = Process(
                        target=self._async_two_pass_cleanup_impl,
                        args=(parent_dir, global_step, save_freq, patience, self.world_size, self.rank)
                    )
                    self._cleanup_process.start()
                    process_time = time.time() - process_start
                    logger.info(f"Rank {self.rank} [CLEANUP] Async cleanup process started in {process_time:.2f}s, PID: {self._cleanup_process.pid}")
                else:
                    # Synchronous cleanup (current behavior)
                    sync_cleanup_start = time.time()
                    self.two_pass_cleanup(parent_dir, global_step, save_freq, patience)
                    sync_cleanup_time = time.time() - sync_cleanup_start
                    logger.info(f"Rank {self.rank} [CLEANUP] Synchronous cleanup completed in {sync_cleanup_time:.2f}s")
            
            # Only barrier if synchronous cleanup was performed
            if not self.async_cleanup:
                barrier_start = time.time()
                torch.distributed.barrier()
                barrier_time = time.time() - barrier_start
                if barrier_time > 1.0:
                    logger.info(f"Rank {self.rank} [CLEANUP] Post-cleanup barrier wait time: {barrier_time:.2f}s")
            
            cleanup_total_time = time.time() - cleanup_start
            logger.info(f"Rank {self.rank} [CLEANUP] Total cleanup overhead: {cleanup_total_time:.2f}s")
        
        save_checkpoint_total_time = time.time() - save_checkpoint_start
        logger.info(f"Rank {self.rank} [SAVE_CHECKPOINT_COMPLETE] Total checkpoint save time: {save_checkpoint_total_time:.2f}s")
