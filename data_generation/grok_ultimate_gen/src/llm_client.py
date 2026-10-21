from __future__ import annotations

import os
import re
import time
from typing import Any

from .io_utils import safe_json_loads
from .prompts import fact_extraction_prompt, generation_prompt, retry_generation_prompt


class LLMClient:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.llm_config = config["llm"]
        client_type = self.llm_config.get("client_type", "openai_compatible_chat")
        if client_type != "openai_compatible_chat":
            raise NotImplementedError(f"Unsupported llm.client_type: {client_type}")
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("Install `openai` to use LLM API steps.") from exc

        api_key_env = self.llm_config.get("api_key_env", "XAI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.llm_config.get("base_url"),
            timeout=float(self.llm_config.get("timeout_seconds", 3600)),
        )

    def call_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        max_retries = int(self.llm_config.get("max_retries_api", 5))
        sleep_base = float(self.llm_config.get("sleep_between_calls_seconds", 0.0))
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if top_p is not None:
                    kwargs["top_p"] = top_p
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                return text, _usage_to_dict(getattr(response, "usage", None))
            except Exception as exc:
                last_error = exc
                if attempt == max_retries - 1:
                    break
                delay = max(sleep_base, 1.0) * (2**attempt)
                time.sleep(delay)
        raise RuntimeError(f"LLM chat call failed after {max_retries} retries: {last_error}") from last_error

    def extract_fact_card_llm(self, article_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
        max_retries = int(self.llm_config.get("max_retries_api", 5))
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                text, usage = self.call_chat(
                    messages=fact_extraction_prompt(article_text, self.config),
                    model=self.llm_config["fact_model"],
                    temperature=float(self.llm_config.get("temperature_fact", 0.0)),
                    max_tokens=int(self.llm_config.get("max_output_tokens_fact", 1600)),
                )
                card = safe_json_loads(text)
                if not isinstance(card, dict):
                    raise ValueError("Fact extraction response was not a JSON object")
                return card, usage
            except Exception as exc:
                last_error = exc
                if attempt == max_retries - 1:
                    break
                time.sleep(1.0 * (2**attempt))
        raise RuntimeError(f"Fact card extraction failed: {last_error}") from last_error

    def generate_article_llm(
        self,
        structure_card: dict[str, Any],
        fact_card: dict[str, Any],
        template_id: str,
        fail_reasons: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if fail_reasons:
            messages = retry_generation_prompt(structure_card, fact_card, None, fail_reasons, template_id)
        else:
            messages = generation_prompt(structure_card, fact_card, template_id)
        text, usage = self.call_chat(
            messages=messages,
            model=self.llm_config["generation_model"],
            temperature=float(self.llm_config.get("temperature_generation", 0.7)),
            top_p=float(self.llm_config.get("top_p_generation", 0.9)),
            max_tokens=int(self.llm_config.get("max_output_tokens_generation", 1800)),
        )
        return clean_generated_text(text), usage


def extract_fact_card_llm(article_text: str, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return LLMClient(config).extract_fact_card_llm(article_text)


def generate_article_llm(
    structure_card: dict[str, Any],
    fact_card: dict[str, Any],
    config: dict[str, Any],
    template_id: str,
    fail_reasons: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    return LLMClient(config).generate_article_llm(structure_card, fact_card, template_id, fail_reasons)


def clean_generated_text(text: str) -> str:
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:text|markdown)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    stripped = re.sub(r"^\s*(Article|Output|Draft):\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    output = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
    }
    if usage is None:
        return output
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _get_usage_field(usage, key)
        if value is not None:
            output[key] = value
    details = _get_usage_field(usage, "completion_tokens_details")
    reasoning_tokens = _get_usage_field(details, "reasoning_tokens") if details is not None else None
    if reasoning_tokens is not None:
        output["reasoning_tokens"] = reasoning_tokens
    return output


def _get_usage_field(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
