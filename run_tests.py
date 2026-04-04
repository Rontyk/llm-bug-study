import argparse
import json
import subprocess
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, RESULTS_DIR

TIMEOUT_SEC = 10  # Max execution time per test


def load_tasks_map() -> dict:
    tasks_path = DATA_DIR / "all_tasks.json"
    with open(tasks_path) as f:
        tasks = json.load(f)
    return {t["task_id"]: t for t in tasks}


def load_generations(results_file: str | None = None, model: str | None = None) -> list[dict]:
    if results_file:
        with open(results_file) as f:
            return json.load(f)

    # Load from combined JSONL
    combined = RESULTS_DIR / "all_generations.jsonl"
    if not combined.exists():
        print("No results found. Run run_generation.py first.")
        sys.exit(1)

    results = []
    with open(combined) as f:
        for line in f:
            record = json.loads(line.strip())
            if model and record["model"] != model:
                continue
            results.append(record)
    return results


def clean_generated_code(code: str) -> str:
    if not code:
        return ""
    # Strip markdown code fences
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def build_test_script(generated_code: str, task: dict) -> str:
    code = clean_generated_code(generated_code)

    if task["benchmark"] == "humaneval":
        # HumanEval: prompt is function signature, generated is the body
        # The test code uses check() function
        script = f"""\
{task['prompt']}{code}

{task['test_code']}

check({task['entry_point']})\
{code}

{task['test_code']}Execute a test script in an isolated subprocess."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
        if result.returncode == 0:
            return {
                "status": "PASS",
                "error_type": None,
                "error_message": None,
                "stderr": result.stderr[:500] if result.stderr else None,
            }
        else:
            error_type = classify_error(result.stderr)
            return {
                "status": "FAIL",
                "error_type": error_type,
                "error_message": result.stderr[-1000:] if result.stderr else "Unknown error",
                "stderr": result.stderr[:2000] if result.stderr else None,
            }
    except subprocess.TimeoutExpired:
        return {
            "status": "FAIL",
            "error_type": "TimeoutError",
            "error_message": f"Execution exceeded {TIMEOUT_SEC}s timeout",
            "stderr": None,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "stderr": None,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def classify_error(stderr: str) -> str:
    if not stderr:
        return "UnknownError"
    # Look for the last line that contains an error type
    for line in reversed(stderr.strip().split("\n")):
        line = line.strip()
        if "Error" in line or "Exception" in line:
            # Extract just the error class name
            if ":" in line:
                return line.split(":")[0].strip()
            return line.strip()
    return "UnknownError"


def run_all_tests(
    generations: list[dict],
    tasks_map: dict,
) -> list[dict]:
    tested = []
    pass_count = 0
    total = len(generations)

    print(f"\nTesting {total} generations...")
    print("─" * 50)

    for i, gen in enumerate(generations):
        task_id = gen["task_id"]
        model = gen["model"]

        print(f"  [{i+1}/{total}] {model} / {task_id}...", end=" ", flush=True)

        # Skip if generation failed
        if gen.get("error") or not gen.get("generated_code"):
            result = {
                "status": "SKIP",
                "error_type": "GenerationFailed",
                "error_message": gen.get("error", "No code generated"),
                "stderr": None,
            }
            print("SKIP (no code)")
        else:
            task = tasks_map.get(task_id)
            if not task:
                result = {
                    "status": "ERROR",
                    "error_type": "TaskNotFound",
                    "error_message": f"Task {task_id} not in benchmark data",
                    "stderr": None,
                }
                print("ERROR (task not found)")
            else:
                script = build_test_script(gen["generated_code"], task)
                result = execute_test(script)
                status = result["status"]
                if status == "PASS":
                    pass_count += 1
                    print("PASS")
                else:
                    print(f"FAIL ({result['error_type']})")

        # Merge generation data + test result
        tested.append({**gen, "test_result": result})

    # Summary
    print(f"\n{'='*50}")
    print(f"Results: {pass_count}/{total} PASS ({pass_count/max(total,1)*100:.1f}%)")
    print(f"{'='*50}")

    return tested


def save_test_results(tested: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"tested_results_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(tested, f, indent=2)

    print(f"Saved → {output_path}")

    # Also generate a CSV summary for easy analysis
    csv_path = RESULTS_DIR / f"summary_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("task_id,benchmark,model,status,error_type,latency_sec\n")
        for r in tested:
            tr = r["test_result"]
            f.write(
                f"{r['task_id']},{r['benchmark']},{r['model']},"
                f"{tr['status']},{tr.get('error_type','')},{r.get('latency_sec','')}\n"
            )
    print(f"CSV  → {csv_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="LLM Bug Study — Test Runner")
    parser.add_argument("--model", type=str, help="Test only this model")
    parser.add_argument("--results-file", type=str, help="Specific results JSON")
    args = parser.parse_args()

    tasks_map = load_tasks_map()
    generations = load_generations(args.results_file, args.model)
    tested = run_all_tests(generations, tasks_map)
    save_test_results(tested)


if __name__ == "__main__":
    main()
