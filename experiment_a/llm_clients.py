# llm_clients.py
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import requests


@dataclass
class LLMConfig:
    name: str
    provider: str  # "openai_compat" | "anthropic" | "gemini"
    model: str

    base_url: str = "https://api.openai.com/v1"
    api_key: str = "EMPTY"

    temperature: float = 0.0
    max_tokens: int = 256
    timeout_s: int = 60


def _resolve_api_key(value: str) -> str:
    """
    Supports:
      - "ENV:OPENAI_API_KEY"
      - raw "sk-..."
      - "EMPTY" for local servers
    """
    if not value:
        return ""
    if value.startswith("ENV:"):
        env = value.replace("ENV:", "")
        return os.environ.get(env, "")
    return value


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """Tries strict json.loads; if that fails, extracts the first {...} block."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{text}")


class BaseClient:
    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleClient(BaseClient):
    """
    Works for: OpenAI API, vLLM, LM Studio, OpenRouter, Together, Groq.
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self.api_key = _resolve_api_key(cfg.api_key)
        self._is_openai = "openai.com" in self.base_url

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        url = self.base_url + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.cfg.model,
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._is_openai:
            payload["response_format"] = {"type": "json_object"}

        r = requests.post(url, headers=headers, json=payload, timeout=int(self.cfg.timeout_s))
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return _extract_first_json_object(content)


class AnthropicClaudeClient(BaseClient):
    """Anthropic Messages API: https://api.anthropic.com/v1/messages"""
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = _resolve_api_key(cfg.api_key)

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.cfg.model,
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_tokens),
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        r = requests.post(url, headers=headers, json=payload, timeout=int(self.cfg.timeout_s))
        r.raise_for_status()
        data = r.json()

        blocks = data.get("content", [])
        if not blocks:
            raise ValueError(f"Anthropic response missing content: {data}")
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return _extract_first_json_object(text)


class GeminiClient(BaseClient):
    """
    Google AI Studio (Generative Language API):
      https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent?key=...
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = _resolve_api_key(cfg.api_key)
        model = cfg.model
        if not model.startswith("models/"):
            model = "models/" + model
        self.model = model

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": system + "\n\n" + user}]
            }],
            "generationConfig": {
                "temperature": float(self.cfg.temperature),
                "maxOutputTokens": int(self.cfg.max_tokens),
                "responseMimeType": "application/json",
            }
        }

        r = requests.post(url, json=payload, timeout=int(self.cfg.timeout_s))
        r.raise_for_status()
        data = r.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"Gemini response missing candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise ValueError(f"Gemini response missing parts: {data}")
        return _extract_first_json_object(parts[0].get("text", ""))


def make_client(cfg: LLMConfig) -> BaseClient:
    prov = cfg.provider.lower().strip()
    if prov in ("openai", "openai_compat", "openai-compatible", "openai_compatible"):
        return OpenAICompatibleClient(cfg)
    if prov in ("anthropic", "claude"):
        return AnthropicClaudeClient(cfg)
    if prov in ("gemini", "google"):
        return GeminiClient(cfg)
    raise ValueError(f"Unknown provider '{cfg.provider}' for LLM '{cfg.name}'")


def load_llm_configs(path: str) -> List[LLMConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: List[LLMConfig] = []
    for r in raw:
        out.append(
            LLMConfig(
                name=r["name"],
                provider=r.get("provider", "openai_compat"),
                model=r["model"],
                base_url=r.get("base_url", "https://api.openai.com/v1"),
                api_key=r.get("api_key", "EMPTY"),
                temperature=float(r.get("temperature", 0.0)),
                max_tokens=int(r.get("max_tokens", 256)),
                timeout_s=int(r.get("timeout_s", 60)),
            )
        )
    return out
