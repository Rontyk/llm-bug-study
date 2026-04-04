import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, REQUEST_DELAY_SEC, MAX_RETRIES


@dataclass
class GenerationResult:
    task_id: str
    model_name: str
    generated_code: str
    prompt: str
    latency_sec: float
    input_tokens: int
    output_tokens: int
    raw_response: dict


class BaseLLMClient(ABC):

    def __init__(self, model_name: str, config: dict):
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, task_id: str) -> GenerationResult:
        pass

    def _retry_with_backoff(self, func, *args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)


class GroqKeyPool:

    def __init__(self):
        multi = os.environ.get("GROQ_API_KEYS", "")
        if multi:
            self.keys = [k.strip() for k in multi.split(",") if k.strip()]
        else:
            single = os.environ.get("GROQ_API_KEY", "")
            if not single:
                raise ValueError("Set GROQ_API_KEYS or GROQ_API_KEY env variable")
            self.keys = [single]

        self.index = 0
        print(f"  [GroqKeyPool] Loaded {len(self.keys)} key(s)")

    @property
    def current_key(self) -> str:
        return self.keys[self.index]

    def rotate(self) -> bool:
        if len(self.keys) <= 1:
            return False
        self.index = (self.index + 1) % len(self.keys)
        print(f"  [GroqKeyPool] Rotated to key #{self.index + 1}/{len(self.keys)}")
        return True

    def all_exhausted(self, exhausted: set) -> bool:
        return len(exhausted) >= len(self.keys)


_groq_key_pool: GroqKeyPool | None = None

def get_groq_key_pool() -> GroqKeyPool:
    global _groq_key_pool
    if _groq_key_pool is None:
        _groq_key_pool = GroqKeyPool()
    return _groq_key_pool


class OpenAIClient(BaseLLMClient):

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, prompt: str, task_id: str) -> GenerationResult:
        system_msg = (
            "You are a Python coding assistant. "
            "Complete the given function. Return ONLY the code, "
            "no explanations, no markdown fences."
        )

        start = time.time()
        response = self._retry_with_backoff(
            self.client.chat.completions.create,
            model=self.config["model_id"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )
        latency = time.time() - start

        return GenerationResult(
            task_id=task_id,
            model_name=self.model_name,
            generated_code=response.choices[0].message.content.strip(),
            prompt=prompt,
            latency_sec=round(latency, 3),
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            raw_response=response.model_dump(),
        )


class AnthropicClient(BaseLLMClient):

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, prompt: str, task_id: str) -> GenerationResult:
        system_msg = (
            "You are a Python coding assistant. "
            "Complete the given function. Return ONLY the code, "
            "no explanations, no markdown fences."
        )

        start = time.time()
        response = self._retry_with_backoff(
            self.client.messages.create,
            model=self.config["model_id"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - start

        return GenerationResult(
            task_id=task_id,
            model_name=self.model_name,
            generated_code=response.content[0].text.strip(),
            prompt=prompt,
            latency_sec=round(latency, 3),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_response=response.model_dump(),
        )


class GoogleKeyPool:

    def __init__(self):
        multi = os.environ.get("GOOGLE_API_KEYS", "")
        if multi:
            self.keys = [k.strip() for k in multi.split(",") if k.strip()]
        else:
            single = os.environ.get("GOOGLE_API_KEY", "")
            if not single:
                raise ValueError("Set GOOGLE_API_KEYS or GOOGLE_API_KEY env variable")
            self.keys = [single]

        self.index = 0
        print(f"  [GoogleKeyPool] Loaded {len(self.keys)} key(s)")

    @property
    def current_key(self) -> str:
        return self.keys[self.index]

    def rotate(self) -> bool:
        if len(self.keys) <= 1:
            return False
        self.index = (self.index + 1) % len(self.keys)
        print(f"  [GoogleKeyPool] Rotated to key #{self.index + 1}/{len(self.keys)}")
        return True

    def all_exhausted(self, exhausted: set) -> bool:
        return len(exhausted) >= len(self.keys)


_google_key_pool: GoogleKeyPool | None = None

def get_google_key_pool() -> GoogleKeyPool:
    global _google_key_pool
    if _google_key_pool is None:
        _google_key_pool = GoogleKeyPool()
    return _google_key_pool


class GoogleClient(BaseLLMClient):

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.pool = get_google_key_pool()

    def _make_client(self):
        from google import genai
        return genai.Client(api_key=self.pool.current_key)

    def generate(self, prompt: str, task_id: str) -> GenerationResult:
        from google.genai import types

        system_msg = (
            "You are a Python coding assistant. "
            "Complete the given function. Return ONLY the code, "
            "no explanations, no markdown fences."
        )

        exhausted_keys = set()
        max_attempts = MAX_RETRIES * len(self.pool.keys)

        for attempt in range(max_attempts):
            try:
                client = self._make_client()
                start = time.time()
                response = client.models.generate_content(
                    model=self.config["model_id"],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_msg,
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_tokens"],
                        # Disable thinking to save tokens
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=0,
                        ),
                    ),
                )
                latency = time.time() - start

                return GenerationResult(
                    task_id=task_id,
                    model_name=self.model_name,
                    generated_code=response.text.strip(),
                    prompt=prompt,
                    latency_sec=round(latency, 3),
                    input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
                    output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
                    raw_response={"text": response.text},
                )

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "rate" in err_str.lower()

                if is_rate_limit:
                    exhausted_keys.add(self.pool.index)
                    print(f"  [429] Google key #{self.pool.index + 1} rate-limited.")

                    if self.pool.all_exhausted(exhausted_keys):
                        raise RuntimeError(
                            f"All {len(self.pool.keys)} Google keys exhausted. "
                            "Add more keys (from separate projects) to GOOGLE_API_KEYS or wait for reset."
                        ) from e

                    # Find first non-exhausted key
                    for i in range(len(self.pool.keys)):
                        if i not in exhausted_keys:
                            self.pool.index = i
                            print(f"  [GoogleKeyPool] Switched to key #{i + 1}/{len(self.pool.keys)}")
                            break

                    time.sleep(2)
                    continue

                # Non-rate-limit error: standard backoff
                if attempt >= MAX_RETRIES - 1:
                    raise
                wait = 2 ** (attempt % MAX_RETRIES)
                print(f"  Retry {attempt + 1} after {wait}s: {e}")
                time.sleep(wait)

        raise RuntimeError(f"Failed after {max_attempts} attempts")


class GroqClient(BaseLLMClient):

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.pool = get_groq_key_pool()

    def _make_client(self):
        from openai import OpenAI
        return OpenAI(
            api_key=self.pool.current_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def generate(self, prompt: str, task_id: str) -> GenerationResult:
        system_msg = (
            "You are a Python coding assistant. "
            "Complete the given function. Return ONLY the code, "
            "no explanations, no markdown fences."
        )

        exhausted_keys = set()
        max_attempts = MAX_RETRIES * len(self.pool.keys)

        for attempt in range(max_attempts):
            try:
                client = self._make_client()
                start = time.time()
                response = client.chat.completions.create(
                    model=self.config["model_id"],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                )
                latency = time.time() - start

                return GenerationResult(
                    task_id=task_id,
                    model_name=self.model_name,
                    generated_code=response.choices[0].message.content.strip(),
                    prompt=prompt,
                    latency_sec=round(latency, 3),
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    raw_response=response.model_dump(),
                )

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()

                if is_rate_limit:
                    exhausted_keys.add(self.pool.index)
                    print(f"  [429] Key #{self.pool.index + 1} rate-limited.")

                    if self.pool.all_exhausted(exhausted_keys):
                        raise RuntimeError(
                            f"All {len(self.pool.keys)} Groq keys exhausted. "
                            "Add more keys to GROQ_API_KEYS or wait for reset."
                        ) from e

                    # Find first non-exhausted key
                    for i in range(len(self.pool.keys)):
                        if i not in exhausted_keys:
                            self.pool.index = i
                            print(f"  [GroqKeyPool] Switched to key #{i + 1}/{len(self.pool.keys)}")
                            break

                    time.sleep(1)
                    continue

                # Non-rate-limit error: standard backoff
                if attempt >= MAX_RETRIES - 1:
                    raise
                wait = 2 ** (attempt % MAX_RETRIES)
                print(f"  Retry {attempt + 1} after {wait}s: {e}")
                time.sleep(wait)

        raise RuntimeError(f"Failed after {max_attempts} attempts")


class DeepSeekClient(BaseLLMClient):

    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

    def generate(self, prompt: str, task_id: str) -> GenerationResult:
        system_msg = (
            "You are a Python coding assistant. "
            "Complete the given function. Return ONLY the code, "
            "no explanations, no markdown fences."
        )

        start = time.time()
        response = self._retry_with_backoff(
            self.client.chat.completions.create,
            model=self.config["model_id"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )
        latency = time.time() - start

        return GenerationResult(
            task_id=task_id,
            model_name=self.model_name,
            generated_code=response.choices[0].message.content.strip(),
            prompt=prompt,
            latency_sec=round(latency, 3),
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            raw_response=response.model_dump(),
        )


def get_client(model_name: str) -> BaseLLMClient:
    config = MODELS[model_name]
    provider = config["provider"]

    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleClient,
        "groq": GroqClient,
        "deepseek": DeepSeekClient,
    }

    if provider not in clients:
        raise ValueError(f"Unknown provider: {provider}")

    return clients[provider](model_name, config)