import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from config import DATA_DIR, BENCHMARKS


def load_humaneval() -> list[dict]:
    print("Loading HumanEval...")
    ds = load_dataset(
        BENCHMARKS["humaneval"]["source"],
        split=BENCHMARKS["humaneval"]["split"],
        trust_remote_code=True,
    )

    tasks = []
    for item in ds:
        task = {
            "task_id": item["task_id"],  # e.g. "HumanEval/0"
            "benchmark": "humaneval",
            "prompt": item["prompt"],  # function signature + docstring
            "entry_point": item["entry_point"],
            "test_code": item["test"],
            "canonical_solution": item["canonical_solution"],
        }
        tasks.append(task)

    print(f"  Loaded {len(tasks)} HumanEval tasks")
    return tasks


def load_mbpp_sanitized() -> list[dict]:
    print("Loading MBPP-sanitized...")
    ds = load_dataset(
        BENCHMARKS["mbpp_sanitized"]["source"],
        split="test",
    )

    tasks = []
    for item in ds:
        # Build a prompt similar to HumanEval style
        # MBPP gives: text (description), test_list, code (solution)
        prompt = build_mbpp_prompt(item["text"], item["test_list"])

        task = {
            "task_id": f"mbpp/{item['task_id']}",
            "benchmark": "mbpp_sanitized",
            "prompt": prompt,
            "entry_point": extract_function_name(item["code"]),
            "test_code": "\n".join(item["test_list"]),
            "canonical_solution": item["code"],
        }
        tasks.append(task)

    print(f"  Loaded {len(tasks)} MBPP-sanitized tasks")
    return tasks


def build_mbpp_prompt(description: str, test_list: list[str]) -> str:
    test_examples = "\n".join(test_list[:3])  # Show up to 3 test cases
    return (
        f"Write a Python function that solves the following problem:\n\n"
        f"{description}\n\n"
        f"Your function should pass these test cases:\n"
        f"```python\n{test_examples}\n```\n\n"
        f"Write only the function definition. Do not include test cases or explanations."
    )


def extract_function_name(code: str) -> str:
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("def "):
            name = line.split("(")[0].replace("def ", "")
            return name
    return "solution"


def load_all_tasks() -> list[dict]:
    tasks = []
    tasks.extend(load_humaneval())
    tasks.extend(load_mbpp_sanitized())
    return tasks


def save_tasks(tasks: list[dict], filename: str = "all_tasks.json"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / filename
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Saved {len(tasks)} tasks to {output_path}")
    return output_path


if __name__ == "__main__":
    tasks = load_all_tasks()
    save_tasks(tasks)
    print(f"\nTotal: {len(tasks)} tasks ready for generation")
