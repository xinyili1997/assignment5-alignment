import json
import torch
from typing import Callable, Literal

from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tests.adapters import log_generations, log_generations_with_policy


# Helper functions
def get_gsm8k_train_dataset(index: int) -> dict:
    dataset = load_dataset("openai/gsm8k", "main")
    return dataset["train"][index]


def get_gsm8k_test_dataset(
    indices: list[int] | None = None, num_examples: int = 10
) -> list[dict]:
    """Get validation examples from GSM8K test set."""
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]  # Only train and test

    if indices is None:
        indices = list(range(min(num_examples, len(test_data))))

    return [test_data[i] for i in indices]


def load_prompt_template(file_path) -> str:
    with open(file_path, "r") as f:
        template = f.read()  # Read as single string, not list of lines
    return template


def generate_responses(
    vllm_model: LLM,
    sampling_params: SamplingParams,
    prompt_template: str,
    num_val_examples: int = 10,
) -> tuple[list[str], list[str], list[str]]:
    """Generate responses using vLLM."""
    val_datasets = get_gsm8k_test_dataset(num_examples=num_val_examples)
    prompts = []
    ground_truths = []
    for dataset in val_datasets:
        question = dataset["question"]
        prompt = prompt_template.format(question=question)
        prompts.append(prompt)
        ground_truths.append(dataset["answer"])
    # Generate responses using vLLM
    raw_outputs = vllm_model.generate(prompts, sampling_params)
    generated_responses = [output.outputs[0].text for output in raw_outputs]
    results = {
        "prompts": prompts,
        "ground_truths": ground_truths,
        "generated_responses": generated_responses,
    }
    # Serialize results to disk
    output_path = "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


# GRPO Configuration
class GRPOConfig:
    def __init__(self, config_path: str):
        self.n_grpo_steps: int = 200
        self.learning_rate: float = 1e-5
        self.advantage_eps: float = 1e-6
        self.rollout_batch_size: int = 256
        self.group_size: int = 8
        self.sampling_temperature: float = 1.0
        self.sampling_min_tokens: int = (
            4  # As in Expiter, disallow empty string responses
        )
        self.sampling_max_tokens: int = 1024
        self.epochs_per_rollout_batch: int = 1  # On-policy
        self.train_batch_size: int = 256  # On-policy
        self.gradient_accumulation_steps: int = (
            128  # microbatch size is 2, will fit on H100
        )
        self.gpu_memory_utilization: float = 0.85
        self.loss_type: Literal[
            "no_baseline",
            "reinforce_with_baseline",
            "grpo_clip",
        ] = "reinforce_with_baseline"
        self.use_std_normalization: bool = True

        # Hyperparameters consistency checks
        assert self.train_batch_size % self.gradient_accumulation_steps == 0
        self.micro_train_batch_size = (
            self.train_batch_size // self.gradient_accumulation_steps
        )
        assert self.rollout_batch_size % self.group_size == 0
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size
        assert self.train_batch_size >= self.group_size
        self.n_microbatches_per_rollout_batch = (
            self.rollout_batch_size // self.micro_train_batch_size
        )


class GRPO:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-1.5B"):
        self.config = GRPOConfig(config_path="")
        self.prompt_template = load_prompt_template(
            "cs336_alignment/prompts/r1_zero.prompt"
        )
        self.vllm_model = LLM(model=model_name) # for fast generation
        self.sampling_params = SamplingParams(
            temperature=self.config.sampling_temperature,
            top_p=1.0,
            max_tokens=self.config.sampling_max_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        self.reward_fn = r1_zero_reward_fn
        # Tokenizer for log_generations (assuming same model as vLLM)

        # For training the policy model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",  # Requires flash-attn package
        )
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def train_loop(self) -> None:
        pass

    def eval_loop(
        self, num_val_examples: int = 10, log_every_n_steps: int | None = None
    ) -> None:
        """Evaluation loop with logging of generations."""
        results = generate_responses(
            self.vllm_model,
            self.sampling_params,
            self.prompt_template,
            num_val_examples,
        )

        # Option 2: Using the new version (requires policy_model, always computes entropy)
        log_results = log_generations_with_policy(
            policy_model=self.policy_model,
            tokenizer=self.tokenizer,
            prompts=results["prompts"],
            ground_truths=results["ground_truths"],
            reward_fn=self.reward_fn,
            generated_responses=results["generated_responses"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Print summary
        summary = log_results["summary"]
        print("\n" + "=" * 50)
        print("Generation Logging Summary:")
        print(f"  Num examples: {summary['num_examples']}")
        print(f"  Num correct: {summary['num_correct']}")
        print(f"  Num incorrect: {summary['num_incorrect']}")
        print(f"  Avg response length: {summary['avg_response_length']:.2f}")
        if "avg_token_entropy" in summary:
            print(f"  Avg token entropy: {summary['avg_token_entropy']:.4f}")
        print(
            f"  Avg correct response length: {summary['avg_correct_response_length']:.2f}"
        )
        print(
            f"  Avg incorrect response length: {summary['avg_incorrect_response_length']:.2f}"
        )
        print("=" * 50 + "\n")

        return log_results


def main():
    grpo = GRPO()
    grpo.eval_loop()


if __name__ == "__main__":
    main()
