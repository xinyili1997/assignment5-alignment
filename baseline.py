import json
from typing import Callable

from vllm import LLM, SamplingParams
from datasets import load_dataset
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def get_countdown_dataset(index: int) -> dict:
    # prompt:
    # Use {nums} (each at most once) with + − × ÷ to reach {target}.
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")

    result = {}
    result["nums"] = ",".join(str(n) for n in dataset["train"][index]["nums"])
    result["target"] = str(dataset["train"][index]["target"])
    # print(result)
    return result


def get_gsm8k_dataset(index: int) -> dict:
    dataset = load_dataset("openai/gsm8k", "main")
    return dataset["train"][index]


def load_prompt_template(file_path) -> str:
    with open(file_path, "r") as f:
        template = f.read()  # Read as single string, not list of lines
    return template


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    question_prompt: str,
    ground_truth: str,
    eval_sampling_params: SamplingParams,
) -> list[dict]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and return structured results.
    """
    results = []
    raw_outputs = vllm_model.generate(question_prompt, eval_sampling_params)
    for output in raw_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        eval_metrics = reward_fn(generated_text, ground_truth)
        
        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "eval_metrics": eval_metrics,
        }
        results.append(result)
        
        # print(f"question: {prompt!r}")
        print(f"generated: {generated_text!r}")
        print(f"ground_truth: {ground_truth!r}")
        print(f"eval_metrics: {eval_metrics!r}")
        print("-" * 50)
    
    return results


def main():
    # Sample prompts.
    prompt_template = load_prompt_template("cs336_alignment/prompts/r1_zero.prompt")

    # Create an LLM.
    # llm = LLM(model="gpt2")
    llm =  LLM(model="Qwen/Qwen2.5-Math-1.5B")
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=128, stop=["\n"]
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    
    all_results = []
    for i in range(1):
        dataset = get_gsm8k_dataset(index=i)
        question = dataset["question"]
        prompt = prompt_template.format(question=question)
        # print(f"prompt: {prompt!r}")
        results = evaluate_vllm(
            llm, r1_zero_reward_fn, prompt, dataset["answer"], sampling_params
        )
        all_results.extend(results)
    
    # Serialize results to disk
    output_path = "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
   main()
