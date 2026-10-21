from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


Message = dict[str, str]


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    usage: dict[str, Any]
    raw: Any = None


class ProviderClient(Protocol):
    provider: str
    model: str

    async def generate_text(
        self,
        messages: list[Message],
        *,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> ProviderResponse:
        ...

    async def generate_json(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> tuple[dict[str, Any], ProviderResponse]:
        ...


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = _strip_markdown_fences(text)
    candidates = [stripped]
    embedded = _first_top_level_json_object(stripped)
    if embedded and embedded not in candidates:
        candidates.append(embedded)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    preview = stripped.replace("\n", " ")[:240]
    raise ValueError(f"Could not parse JSON object from provider response: {preview!r}")


def _strip_markdown_fences(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|text|markdown)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _first_top_level_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def split_system_user(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system_parts = [message["content"] for message in messages if message.get("role") == "system"]
    non_system = [message for message in messages if message.get("role") != "system"]
    system = "\n\n".join(part for part in system_parts if part.strip()) or None
    return system, non_system


def messages_to_gemini_contents(messages: list[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not content:
            continue
        if role == "user":
            parts.append(content)
        else:
            parts.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(parts).strip()


def usage_from_attrs(obj: Any, mapping: dict[str, str]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    if obj is None:
        return usage
    for out_key, attr in mapping.items():
        value = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
        usage[out_key] = value
    return usage


class GeminiProviderClient:
    provider = "gemini"

    def __init__(self, model: str, thinking_budget: int | None = 512) -> None:
        load_dotenv_if_available()
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Missing dependency: pip install google-genai") from exc
        self.model = model
        self.thinking_budget = thinking_budget
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"
        if not project:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT for Gemini Vertex AI mode.")
        self.client = genai.Client(vertexai=True, project=project, location=location)

    async def generate_text(
        self,
        messages: list[Message],
        *,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> ProviderResponse:
        return await self._generate(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            response_mime_type=None,
        )

    async def generate_json(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> tuple[dict[str, Any], ProviderResponse]:
        response = await self._generate(
            messages,
            temperature=temperature,
            top_p=None,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            response_mime_type="application/json",
        )
        return parse_json_object(response.text), response

    async def _generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
        thinking_budget: int | None,
        response_mime_type: str | None,
    ) -> ProviderResponse:
        from google.genai import types

        system, non_system = split_system_user(messages)
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        effective_thinking_budget = self.thinking_budget if thinking_budget is None else thinking_budget
        if effective_thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=effective_thinking_budget,
                include_thoughts=False,
            )
        if top_p is not None:
            config_kwargs["top_p"] = top_p
        if system:
            config_kwargs["system_instruction"] = system
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=messages_to_gemini_contents(non_system),
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = getattr(response, "text", None)
        if not text:
            raise ValueError(f"Gemini response had no text: {response!r}")
        usage = usage_from_attrs(
            getattr(response, "usage_metadata", None),
            {
                "prompt_tokens": "prompt_token_count",
                "completion_tokens": "candidates_token_count",
                "reasoning_tokens": "thoughts_token_count",
                "total_tokens": "total_token_count",
            },
        )
        return ProviderResponse(text=text, usage=usage, raw=response)


class ClaudeProviderClient:
    provider = "claude"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        load_dotenv_if_available()
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY or pass --api-key for Claude.")
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Missing dependency: pip install anthropic") from exc
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate_text(
        self,
        messages: list[Message],
        *,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> ProviderResponse:
        system, non_system = split_system_user(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": non_system,
        }
        if system:
            kwargs["system"] = system
        # Anthropic rejects temperature and top_p together for some current
        # Claude models. Keep temperature as the single sampling control.
        response = await self.client.messages.create(**kwargs)
        text_parts = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        text = "".join(text_parts).strip()
        if not text:
            raise ValueError(f"Claude response had no text: {response!r}")
        usage = usage_from_attrs(
            getattr(response, "usage", None),
            {
                "prompt_tokens": "input_tokens",
                "completion_tokens": "output_tokens",
            },
        )
        if usage:
            prompt = usage.get("prompt_tokens") or 0
            completion = usage.get("completion_tokens") or 0
            usage["total_tokens"] = prompt + completion
        return ProviderResponse(text=text, usage=usage, raw=response)

    async def generate_json(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> tuple[dict[str, Any], ProviderResponse]:
        response = await self.generate_text(
            messages,
            temperature=temperature,
            top_p=None,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        return parse_json_object(response.text), response


class OpenAIProviderClient:
    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 600,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        load_dotenv_if_available()
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or pass --api-key for OpenAI.")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Missing dependency: pip install openai") from exc
        self.model = model
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**kwargs)
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.extra_body = extra_body

    async def generate_text(
        self,
        messages: list[Message],
        *,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> ProviderResponse:
        return await self._generate(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=None,
        )

    async def generate_json(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> tuple[dict[str, Any], ProviderResponse]:
        response = await self._generate(
            messages,
            temperature=temperature,
            top_p=None,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return parse_json_object(response.text), response

    async def _generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
        response_format: dict[str, str] | None,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if not openai_default_sampling_only(self.model):
            kwargs["temperature"] = temperature
        if top_p is not None and not openai_default_sampling_only(self.model):
            kwargs["top_p"] = top_p
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.verbosity:
            kwargs["verbosity"] = self.verbosity
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        attempts = 0
        while True:
            attempts += 1
            try:
                response = await self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                if attempts > 4:
                    raise
                if openai_unsupported_parameter(exc, "max_completion_tokens"):
                    legacy_kwargs = dict(kwargs)
                    legacy_kwargs["max_tokens"] = legacy_kwargs.pop("max_completion_tokens")
                    kwargs = legacy_kwargs
                    continue
                removed = False
                for param_name in ("reasoning_effort", "verbosity"):
                    if param_name in kwargs and openai_unsupported_parameter(exc, param_name):
                        kwargs.pop(param_name, None)
                        removed = True
                        break
                if removed:
                    continue
                for param_name in ("temperature", "top_p"):
                    if param_name in kwargs and openai_rejected_parameter_value(exc, param_name):
                        kwargs.pop(param_name, None)
                        removed = True
                        break
                if removed:
                    continue
                raise
        choices = getattr(response, "choices", []) or []
        if not choices:
            raise ValueError(f"OpenAI chat response had no choices: {response!r}")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        text = str(content or "").strip()
        if not text:
            raise ValueError(f"OpenAI chat response had no text: {response!r}")
        usage = openai_usage(getattr(response, "usage", None))
        return ProviderResponse(text=text, usage=usage, raw=response)


class DeepSeekProviderClient(OpenAIProviderClient):
    provider = "deepseek"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: int = 600,
        thinking: str | None = "disabled",
        reasoning_effort: str | None = None,
    ) -> None:
        load_dotenv_if_available()
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY or pass --api-key for DeepSeek.")
        extra_body = {"thinking": {"type": thinking}} if thinking else None
        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        )


def openai_usage(obj: Any) -> dict[str, Any]:
    usage = usage_from_attrs(
        obj,
        {
            "prompt_tokens": "prompt_tokens",
            "completion_tokens": "completion_tokens",
            "total_tokens": "total_tokens",
        },
    )
    details = getattr(obj, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
    if reasoning_tokens is not None:
        usage["reasoning_tokens"] = reasoning_tokens
    return {key: value for key, value in usage.items() if value is not None}


def openai_unsupported_parameter(exc: Exception, param_name: str) -> bool:
    text = str(exc).lower()
    return "unsupported" in text and param_name.lower() in text


def openai_rejected_parameter_value(exc: Exception, param_name: str) -> bool:
    text = str(exc).lower()
    return ("unsupported value" in text or "invalid" in text) and param_name.lower() in text


def openai_default_sampling_only(model: str) -> bool:
    return model.strip().lower().startswith("gpt-5")


@dataclass
class QwenProviderClient:
    model: str
    api_base: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    timeout_seconds: int = 600
    chat_template_kwargs: dict[str, Any] | None = None

    provider: str = "qwen"

    async def check_server(self) -> None:
        try:
            await asyncio.to_thread(
                _get_json,
                self.api_base,
                "/models",
                self.api_key,
                min(self.timeout_seconds, 30),
            )
        except Exception as exc:
            raise RuntimeError(
                f"vLLM server is not reachable at {self.api_base}. Start `vllm serve ...` before generation."
            ) from exc

    async def generate_text(
        self,
        messages: list[Message],
        *,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        response = await asyncio.to_thread(
            _post_json,
            self.api_base,
            "/chat/completions",
            payload,
            self.api_key,
            self.timeout_seconds,
        )
        choices = response.get("choices")
        if not choices:
            raise ValueError(f"vLLM chat response had no choices: {response!r}")
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, list):
            content = "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        if not content:
            raise ValueError(f"vLLM chat response had no content: {response!r}")
        return ProviderResponse(text=str(content), usage=response.get("usage", {}), raw=response)

    async def generate_json(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> tuple[dict[str, Any], ProviderResponse]:
        response = await self.generate_text(
            messages,
            temperature=temperature,
            top_p=None,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        return parse_json_object(response.text), response


def _endpoint_url(api_base: str, endpoint: str) -> str:
    return api_base.rstrip("/") + endpoint


def _post_json(api_base: str, endpoint: str, payload: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    url = _endpoint_url(api_base, endpoint)
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc
    return _loads_response_json(url, body)


def _get_json(api_base: str, endpoint: str, api_key: str, timeout: int) -> dict[str, Any]:
    url = _endpoint_url(api_base, endpoint)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc
    return _loads_response_json(url, body)


def _loads_response_json(url: str, body: str) -> dict[str, Any]:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {body[:1200]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from {url}: {body[:1200]}")
    return parsed


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv()
    load_dotenv(repo_root / "api_keys.env", override=False)
