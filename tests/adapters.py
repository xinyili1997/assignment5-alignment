from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    # Tokenize prompts to get their lengths (no special tokens for accurate length)
    prompt_tokens = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_tokens = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    combined_tokens = [p + o for p, o in zip(prompt_tokens, output_tokens)]

    # Pad to max length
    max_len = max(len(tokens) for tokens in combined_tokens)
    padded = []
    attention_mask = []
    for i in range(len(combined_tokens)):
        tokens = combined_tokens[i]
        prompt = prompt_tokens[i]
        pad_len = max_len - len(tokens)
        padded.append(tokens + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append(
            [0] * len(prompt) + [1] * (len(tokens) - len(prompt)) + [0] * pad_len
        )

    # Convert to tensors
    padded_tensor = torch.tensor(padded, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.bool)

    # Slice off last token for input_ids, first token for labels
    # Also slice response_mask to match (use labels-aligned version, i.e., slice first)
    # print(f"combined_tokens['input_ids'].shape: {padded_tensor.shape}")
    # print(f"tokens['input_ids'][:, :-1].shape: {padded_tensor[:, :-1].shape}")
    result = {
        "input_ids": padded_tensor[:, :-1],
        "labels": padded_tensor[:, 1:],
        "response_mask": attention_mask_tensor[:, 1:],  # Align with labels
    }
    return result


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # H(p) = - sum(p * log(p)) p = softmax(logits)
    # logits:[batch_size, sequence_length, vocab_size]
    p = torch.softmax(logits, dim=-1)
    return -torch.sum(p * torch.log(p), dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    # log(softmax(model(input_ids))) select the label-th element
    logits = model(input_ids).logits
    p = torch.softmax(logits, dim=-1)
    # p [batch_size, sequence_length, vocab_size]
    # labels [batch_size, sequence_length]
    # gather the labels-th element of the last dimension of p
    log_probs = torch.log(
        torch.gather(p, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    )

    result = {}
    result["log_probs"] = log_probs
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            per-token log-probabilities from the SFT policy being trained.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        normalize_constant: int | None, the constant by which to divide the sum. It is fine to leave this as 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """
    # SFT loss: negative log-likelihood (cross-entropy)
    loss_scale = 1.0 / gradient_accumulation_steps

    # Compute forward pass once (the loss value is the same in each iteration)
    masked_sum = masked_normalize(
        policy_log_probs,
        response_mask,
        dim=None,
        normalize_constant=normalize_constant,
    )
    loss = -masked_sum * loss_scale
    # Accumulate gradients: call backward() gradient_accumulation_steps times
    # Gradients automatically accumulate in policy_log_probs.grad
    # Use retain_graph=True to keep the computation graph for multiple backward() calls
    for _ in range(gradient_accumulation_steps - 1):
        loss = loss * loss_scale
        loss.backward(retain_graph=True)

    # Return the loss adjusted for gradient accumulation (the scaled loss used in backward())
    # This represents the effective loss per update step, useful for logging
    return loss, {"policy_log_probs_grad": policy_log_probs.grad}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    # Sum over a dimension and normalize by a constant,
    # considering only those elements where mask == 1.
    # Multiply by mask to zero out masked elements while preserving shape
    masked_tensor = tensor * mask
    if dim is None:
        # sum over all dimensions
        return masked_tensor.sum() / normalize_constant
    else:
        return masked_tensor.sum(dim=dim) / normalize_constant


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError


# ------------------------------------------------------------
# GPRO.
# ------------------------------------------------------------


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for each group of rollout responses,
    normalized by the group size.

       Solution 1: A = R_i - mean(R_i)
       Solution 2: A = (R_i - mean(R_i)) / (std(R_i) + advantage_eps)

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        rewards.append(reward_fn(response, ground_truth)["reward"])
    # print(f"rewards: {rewards}")
    rewards = torch.tensor(rewards)

    # Reshape rewards to (n_prompts_per_rollout_batch, group_size) for group-wise operations
    rewards_2d = rewards.view(
        -1, group_size
    )  # Shape: (n_prompts_per_rollout_batch, group_size)

    # Compute mean within each group
    group_means = rewards_2d.mean(
        dim=1, keepdim=True
    )  # Shape: (n_prompts_per_rollout_batch, 1)

    if not normalize_by_std:
        # Advantages = rewards - mean(rewards within group)
        advantages_2d = rewards_2d - group_means
    else:
        # Compute std within each group
        group_stds = rewards_2d.std(
            dim=1, keepdim=True
        )  # Shape: (n_prompts_per_rollout_batch, 1)
        # Advantages = (rewards - mean) / (std + eps)
        advantages_2d = (rewards_2d - group_means) / (group_stds + advantage_eps)

    # Reshape back to original shape
    advantages = advantages_2d.view(-1)  # Shape: (rollout_batch_size,)

    return [
        advantages,
        rewards,
        {
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
        },
    ]


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    log_prob * advantage is the "surrogate objective" whose gradient pushes
    probability up for actions with positive advantage and down for negative
    advantage.

    log_prob: indicates the direction to increase the probability of the action.
    If A > 0: action was better than expected → increase its probability.
    If A < 0: action was worse than expected → decrease its probability.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """
    broadcasted_rewards_or_advantages = raw_rewards_or_advantages.expand_as(
        policy_log_probs
    )
    result = -broadcasted_rewards_or_advantages * policy_log_probs
    print(f"result: {result}")
    return result


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Ratio: policy_log_probs / old_log_probs
    Clip:
        if 1-cliprange < ratio < 1+cliprange, keep the ratio as is.
        if ratio < 1-cliprange, set the ratio to 1-cliprange.
        if ratio > 1+cliprange, set the ratio to 1+cliprange.
    Loss: - min(ratio * advantages, clip(ratio, cliprange) * advantages)

    Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length):
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss
                (used to compute clip fraction).
    """
    # Ratio = exp(policy_log_probs - old_log_probs) = exp(policy) / exp(old)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    advantages_broadcasted = advantages.expand_as(ratio)
    # print(f"ratio: {ratio}")
    # print(f"clipped_ratio: {clipped_ratio}")
    # print(f"advantages_broadcasted: {advantages_broadcasted}")
    loss = -torch.min(
        ratio * advantages_broadcasted, clipped_ratio * advantages_broadcasted
    )
    # print(f"loss: {loss}")
    return loss, {"ratio": ratio, "clipped_ratio": clipped_ratio}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        # Advantage = raw_rewards
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise NotImplementedError


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    # Sum the masked elements
    masked_sum = (tensor * mask).sum(dim=dim)
    # Count the masked elements (where mask == 1)
    mask_count = mask.sum(dim=dim)
    # Mean = sum / count (only over masked elements)
    return masked_sum / mask_count


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio.
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    masked_loss = masked_mean(per_token_loss, response_mask, dim=None)
    loss_scale = 1.0 / gradient_accumulation_steps
    loss = masked_loss * loss_scale

    for _ in range(gradient_accumulation_steps - 1):
        loss.backward(retain_graph=True)
    return loss, metadata
