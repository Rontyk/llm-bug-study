import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, RESULTS_DIR, DATA_DIR, REQUEST_DELAY_SEC
from scripts.llm_client import get_client, GenerationResult


def load_tasks(benchmark_filter: str | None = None) -> list[dict]:
    tasks_path = DATA_DIR / "all_tasks.json"
    if not tasks_path.exists():
        print(f"Tasks file not found at {tasks_path}. Run load_benchmarks.py first.")
        sys.exit(1)

    with open(tasks_path) as f:
        tasks = json.load(f)

    if benchmark_filter:
        tasks = [t for t in tasks if t["benchmark"] == benchmark_filter]
        print(f"Filtered to {len(tasks)} tasks from '{benchmark_filter}'")

    return tasks


def load_completed_tasks(model_name: str) -> set[str]:
    combined_path = RESULTS_DIR / "all_generations.jsonl"
    if not combined_path.exists():
        return set()

    completed = set()
    with open(combined_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("model") == model_name and record.get("error") is None:
                    completed.add(record["task_id"])
            except json.JSONDecodeError:
                continue

    return completed


def run_generation(
    model_name: str,
    tasks: list[dict],
    dry_run: bool = False,
    resume: bool = False,
) -> list[dict]:
    client = get_client(model_name)
    results = []
    task_list = tasks[:3] if dry_run else tasks

    # Resume: skip already completed tasks
    if resume:
        completed = load_completed_tasks(model_name)
        if completed:
            original_count = len(task_list)
            task_list = [t for t in task_list if t["task_id"] not in completed]
            print(f"  [Resume] Skipping {original_count - len(task_list)} completed tasks, "
                  f"{len(task_list)} remaining.")
        else:
            print(f"  [Resume] No completed tasks found, starting fresh.")

    # Prepare incremental save file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = RESULTS_DIR / "all_generations.jsonl"

    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Tasks: {len(task_list)}")
    print(f"{'='*60}")

    for i, task in enumerate(task_list):
        task_id = task["task_id"]
        print(f"  [{i+1}/{len(task_list)}] {task_id}...", end=" ", flush=True)

        try:
            result = client.generate(prompt=task["prompt"], task_id=task_id)
            record = {
                "task_id": task_id,
                "benchmark": task["benchmark"],
                "model": model_name,
                "generated_code": result.generated_code,
                "prompt": result.prompt,
                "latency_sec": result.latency_sec,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "error": None,
                "timestamp": datetime.now().isoformat(),
            }
            print(f"OK ({result.latency_sec}s, {result.output_tokens} tokens)")

        except Exception as e:
            record = {
                "task_id": task_id,
                "benchmark": task["benchmark"],
                "model": model_name,
                "generated_code": None,
                "prompt": task["prompt"],
                "latency_sec": None,
                "input_tokens": None,
                "output_tokens": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            print(f"FAILED: {e}")

        results.append(record)

        # Save IMMEDIATELY to jsonl (so data isn't lost on crash/429)
        with open(combined_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Rate limiting
        if i < len(task_list) - 1:
            time.sleep(REQUEST_DELAY_SEC)

    return results


def save_results(results: list[dict], model_name: str, tag: str = ""):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    filename = f"generations_{model_name}{suffix}_{timestamp}.json"
    output_path = RESULTS_DIR / filename

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results → {output_path}")
    return output_path


def print_summary(results: list[dict]):
    total = len(results)
    errors = sum(1 for r in results if r["error"])
    success = total - errors
    avg_latency = sum(
        r["latency_sec"] for r in results if r["latency_sec"]
    ) / max(success, 1)

    print(f"\n{'─'*40}")
    print(f"Summary: {success}/{total} successful ({errors} errors)")
    print(f"Avg latency: {avg_latency:.2f}s")
    print(f"{'─'*40}")


def main():
    parser = argparse.ArgumentParser(description="LLM Bug Study — Code Generation")
    parser.add_argument("--model", type=str, help="Run only this model")
    parser.add_argument("--benchmark", type=str, help="Run only this benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Test with 3 tasks")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks already completed (no error) in all_generations.jsonl")
    args = parser.parse_args()

    tasks = load_tasks(args.benchmark)
    models = [args.model] if args.model else list(MODELS.keys())

    all_results = []
    for model_name in models:
        results = run_generation(
            model_name, tasks,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        save_results(results, model_name, tag="dryrun" if args.dry_run else "")
        print_summary(results)
        all_results.extend(results)

    # Final summary
    print(f"\n{'='*60}")
    print(f"DONE — {len(all_results)} total generations across {len(models)} models")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
