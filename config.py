from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Three scales via Groq (free tier) + DeepSeek V3 via DeepSeek API
MODELS = {
    "llama-3.1-8b": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "llama-3.3-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "gpt-oss-120b": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "deepseek-v3": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
}

BENCHMARKS = {
    "humaneval": {
        "source": "openai/openai_humaneval",  # HuggingFace dataset ID
        "split": "test",
        "n_tasks": 164,
    },
    "mbpp_sanitized": {
        "source": "google-research-datasets/mbpp",
        "split": "sanitized",  # Use the sanitized subset
        "n_tasks": 427,
    },
}

SAMPLES_PER_TASK = 1  # pass@1
REQUEST_DELAY_SEC = 3.0  # Groq free tier has strict rate limits
MAX_RETRIES = 3

# Set in PowerShell:
#   $env:GROQ_API_KEYS="gsk_key1,gsk_key2,..."
#   $env:DEEPSEEK_API_KEY="sk-..."
