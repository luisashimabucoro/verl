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
"""
Training samples tracker for logging per-sample metrics during training.
"""

import json
import os
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss


class TrainingSamplesTracker:
    """Tracker for logging per-sample training metrics (index, accuracy, entropy, generations) for each epoch."""

    def __init__(
        self,
        tokenizer=None,
        log_dir: Optional[str] = None,
        enabled: bool = True,
        n_samples_per_prompt: Optional[int] = None,
    ):
        """
        Initialize the training samples tracker.

        Args:
            tokenizer: Tokenizer for decoding prompts and responses. If None, generations won't be stored.
            log_dir: Directory to save per-sample logs. If None, uses current directory.
            enabled: Whether tracking is enabled.
            n_samples_per_prompt: Number of responses generated per prompt. If None, will attempt to detect from batch structure.
        """
        self.enabled = enabled
        self.tokenizer = tokenizer
        self.log_dir = log_dir or os.getcwd()
        os.makedirs(self.log_dir, exist_ok=True)
        self.n_samples_per_prompt = n_samples_per_prompt
        
        # Track sample index across epochs
        self.sample_index_counter = 0
        self.epoch_samples = defaultdict(list)
        
        # Track prompts separately with their index
        self.prompt_index_counter = 0
        self.prompt_index_map = {}  # Maps (epoch, prompt_hash) -> prompt_index
        self.prompts_by_index = {}  # Maps prompt_index -> prompt_text
        
    def log_samples(
        self,
        batch: DataProto,
        epoch: int,
        global_step: int,
        loss_agg_mode: str = "token-mean",
    ):
        """
        Log per-sample metrics for a batch.

        Args:
            batch: DataProto containing batch data with rewards and entropy.
            epoch: Current epoch number.
            global_step: Current global training step.
            loss_agg_mode: Mode for aggregating loss/entropy (e.g., "token-mean").
        """
        if not self.enabled:
            return

        batch_size = batch.batch.batch_size[0]
        
        # Extract prompts and determine n_samples_per_prompt if not set
        prompts_per_sample = None
        if self.tokenizer is not None and "prompts" in batch.batch.keys():
            prompts = batch.batch["prompts"]  # (bsz, prompt_length)
            
            prompts_per_sample = []
            for i in range(batch_size):
                prompt_ids = prompts[i]
                # Prompts are stored separately, decode directly
                prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                prompts_per_sample.append(prompt_str)
            
            # Detect n_samples_per_prompt if not set by checking if prompts repeat
            if self.n_samples_per_prompt is None and batch_size > 1:
                # Check if first few prompts are identical (indicating multiple responses per prompt)
                first_prompt = prompts_per_sample[0]
                n_samples_detected = 1
                for i in range(1, min(batch_size, 10)):  # Check up to 10 samples
                    if prompts_per_sample[i] == first_prompt:
                        n_samples_detected += 1
                    else:
                        break
                # Only use detected value if it makes sense (batch_size is divisible by it)
                if n_samples_detected > 1 and batch_size % n_samples_detected == 0:
                    self.n_samples_per_prompt = n_samples_detected
                else:
                    self.n_samples_per_prompt = 1
            elif self.n_samples_per_prompt is None:
                self.n_samples_per_prompt = 1
        
        # Extract per-sample accuracy (reward/score)
        accuracy_per_sample = None
        if "token_level_rewards" in batch.batch.keys():
            # Sum token-level rewards to get per-sample reward
            token_level_rewards = batch.batch["token_level_rewards"]  # (bsz, seq_len)
            response_mask = batch.batch.get("response_mask", None)
            if response_mask is not None:
                # Mask out padding tokens
                masked_rewards = token_level_rewards * response_mask.to(token_level_rewards.dtype)
                accuracy_per_sample = masked_rewards.sum(dim=-1).detach().cpu().numpy()  # (bsz,)
            else:
                accuracy_per_sample = token_level_rewards.sum(dim=-1).detach().cpu().numpy()
        elif "token_level_scores" in batch.batch.keys():
            # Use scores if rewards not available
            token_level_scores = batch.batch["token_level_scores"]  # (bsz, seq_len)
            response_mask = batch.batch.get("response_mask", None)
            if response_mask is not None:
                masked_scores = token_level_scores * response_mask.to(token_level_scores.dtype)
                accuracy_per_sample = masked_scores.sum(dim=-1).detach().cpu().numpy()
            else:
                accuracy_per_sample = token_level_scores.sum(dim=-1).detach().cpu().numpy()
        
        # Extract per-sample entropy
        entropy_per_sample = None
        if "entropys" in batch.batch.keys():
            entropys = batch.batch["entropys"]  # (bsz, seq_len)
            response_mask = batch.batch.get("response_mask", None)
            if response_mask is not None:
                # Aggregate entropy per sample using the same mode as loss aggregation
                entropy_per_sample = []
                for i in range(batch_size):
                    sample_entropy = entropys[i:i+1]  # (1, seq_len)
                    sample_mask = response_mask[i:i+1]  # (1, seq_len)
                    sample_entropy_agg = agg_loss(
                        loss_mat=sample_entropy,
                        loss_mask=sample_mask,
                        loss_agg_mode=loss_agg_mode
                    )
                    entropy_per_sample.append(sample_entropy_agg.detach().cpu().item())
                entropy_per_sample = np.array(entropy_per_sample)
            else:
                # If no mask, use mean across sequence length
                entropy_per_sample = entropys.mean(dim=-1).detach().cpu().numpy()
        
        # Extract generations (only responses, not prompts to avoid duplication across epochs)
        responses_per_sample = None
        if self.tokenizer is not None and "responses" in batch.batch.keys():
            responses = batch.batch["responses"]  # (bsz, response_len)
            response_mask = batch.batch.get("response_mask", None)
            
            responses_per_sample = []
            
            for i in range(batch_size):
                # Decode response (only valid tokens based on response_mask)
                response_ids = responses[i]
                if response_mask is not None:
                    valid_response_length = response_mask[i].sum().item()
                    if valid_response_length > 0:
                        valid_response_ids = response_ids[:valid_response_length]
                        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    else:
                        response_str = ""
                else:
                    response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                responses_per_sample.append(response_str)
        
        # Log each sample in the batch
        for i in range(batch_size):
            # Get or create prompt index
            prompt_index = None
            if prompts_per_sample is not None:
                prompt_text = prompts_per_sample[i]
                # Create a hash of prompt for deduplication within epoch
                prompt_hash = hash(prompt_text)
                prompt_key = (epoch, prompt_hash)
                
                if prompt_key not in self.prompt_index_map:
                    # New prompt, assign new index
                    prompt_index = self.prompt_index_counter
                    self.prompt_index_map[prompt_key] = prompt_index
                    self.prompts_by_index[prompt_index] = prompt_text
                    self.prompt_index_counter += 1
                else:
                    # Existing prompt, reuse index
                    prompt_index = self.prompt_index_map[prompt_key]
            
            sample_data = {
                "epoch": epoch,
                "global_step": global_step,
                "sample_index": self.sample_index_counter,
                "batch_index": i,
                "prompt_index": prompt_index,
                "accuracy": float(accuracy_per_sample[i]) if accuracy_per_sample is not None else None,
                "entropy": float(entropy_per_sample[i]) if entropy_per_sample is not None else None,
            }
            
            # Add response generation if available
            if responses_per_sample is not None:
                sample_data["response"] = responses_per_sample[i]
            
            self.epoch_samples[epoch].append(sample_data)
            self.sample_index_counter += 1
    
    def flush_epoch(self, epoch: int):
        """
        Flush and save logs for a completed epoch.

        Args:
            epoch: Epoch number to flush.
        """
        if not self.enabled:
            return

        if epoch not in self.epoch_samples:
            return

        # Collect all unique prompt indices used in this epoch
        prompt_indices_used = set()
        for sample_data in self.epoch_samples[epoch]:
            if "prompt_index" in sample_data and sample_data["prompt_index"] is not None:
                prompt_indices_used.add(sample_data["prompt_index"])
        
        # Save prompts to a separate file, indexed by prompt_index
        prompts_file = os.path.join(self.log_dir, f"prompts_step_{epoch}.jsonl")
        with open(prompts_file, "w") as f:
            for prompt_index in sorted(prompt_indices_used):
                if prompt_index in self.prompts_by_index:
                    prompt_data = {
                        "prompt_index": prompt_index,
                        "prompt": self.prompts_by_index[prompt_index],
                    }
                    f.write(json.dumps(prompt_data) + "\n")
        
        # Group samples by prompt_index to collect all responses per prompt
        samples_by_prompt = defaultdict(list)
        for sample_data in self.epoch_samples[epoch]:
            prompt_idx = sample_data.get("prompt_index")
            if prompt_idx is not None:
                samples_by_prompt[prompt_idx].append(sample_data)
        
        # Save epoch metrics data to JSON file (with prompt_index reference, not prompt text)
        log_file = os.path.join(self.log_dir, f"training_samples_step_{epoch}.jsonl")
        with open(log_file, "w") as f:
            for sample_data in self.epoch_samples[epoch]:
                # Create a copy without the prompt text (only keep prompt_index)
                output_data = {
                    "epoch": sample_data["epoch"],
                    "global_step": sample_data["global_step"],
                    "sample_index": sample_data["sample_index"],
                    "batch_index": sample_data["batch_index"],
                    "prompt_index": sample_data.get("prompt_index"),
                    "accuracy": sample_data.get("accuracy"),
                    "entropy": sample_data.get("entropy"),
                }
                # Add response if available
                if "response" in sample_data:
                    output_data["response"] = sample_data["response"]
                f.write(json.dumps(output_data) + "\n")
        
        # Save generations grouped by prompt (all responses per prompt)
        generations_file = os.path.join(self.log_dir, f"generations_step_{epoch}.jsonl")
        with open(generations_file, "w") as f:
            for prompt_index in sorted(samples_by_prompt.keys()):
                prompt_samples = samples_by_prompt[prompt_index]
                # Collect all responses for this prompt
                responses = []
                for sample in prompt_samples:
                    if "response" in sample:
                        response_data = {
                            "sample_index": sample["sample_index"],
                            "batch_index": sample["batch_index"],
                            "accuracy": sample.get("accuracy"),
                            "entropy": sample.get("entropy"),
                            "response": sample["response"],
                        }
                        responses.append(response_data)
                
                generation_data = {
                    "prompt_index": prompt_index,
                    "epoch": prompt_samples[0]["epoch"] if prompt_samples else epoch,
                    "global_step": prompt_samples[0]["global_step"] if prompt_samples else None,
                    "num_responses": len(responses),
                    "responses": responses,
                }
                f.write(json.dumps(generation_data) + "\n")
        
        # Clean up prompt mappings for this epoch (keep prompts_by_index for potential reuse)
        # Remove prompt_index_map entries for this epoch
        keys_to_remove = [key for key in self.prompt_index_map.keys() if key[0] == epoch]
        for key in keys_to_remove:
            del self.prompt_index_map[key]
        
        # Clear epoch data from memory
        del self.epoch_samples[epoch]
    
    def flush_all(self):
        """Flush all remaining epoch data."""
        if not self.enabled:
            return

        for epoch in list(self.epoch_samples.keys()):
            self.flush_epoch(epoch)

