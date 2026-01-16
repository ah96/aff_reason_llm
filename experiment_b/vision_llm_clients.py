import os
import json
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class LLMConfig:
    """
    Provider-agnostic LLM configuration.

    Expected JSON fields (see llms_updated.json for an example):
      - name: human-readable name used for filenames / tables
      - provider: "openai_compat" | "openai" | "anthropic" | "gemini"
      - model: provider model id
      - api_key: literal key OR "ENV:VARNAME"
      - base_url: (openai/openai_compat only) e.g. "https://api.openai.com/v1" or "http://localhost:8000/v1"
      - temperature: float (default 0.0)
      - max_tokens: int (default 256)
      - supports_vision: bool (default True). If False, the runner will skip this model for image inputs.
    """
    name: str
    provider: str
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_tokens: int = 256
    supports_vision: bool = True


def _resolve_api_key(k: str) -> str:
    if (k or "").startswith("ENV:"):
        return os.environ.get(k.replace("ENV:", ""), "")
    return k


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction: accept exact JSON or a response that contains a JSON object somewhere.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        s = text.find("{")
        e = text.rfind("}")
        if s >= 0 and e > s:
            return json.loads(text[s : e + 1])
        raise ValueError(f"LLM did not return JSON. Output:\n{text}")


class VisionLLMClient:
    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAICompatVisionClient(VisionLLMClient):
    """
    OpenAI Chat Completions-compatible client (works for OpenAI and most OpenAI-compatible servers, e.g. vLLM).

    We intentionally use /chat/completions rather than /responses because:
      - it is widely supported by OpenAI-compatible servers
      - it supports vision via message content with image_url for OpenAI models
    """
    def __init__(self, model: str, api_key: str, base_url: str, temperature: float = 0.0, max_tokens: int = 256):
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def complete_json(self, system: str, user: str, images_b64png: list[str]) -> dict:
        import requests, json

        # IMPORTANT: OpenAI expects a "data:" URL for base64 images
        parts = [{"type": "text", "text": user}]
        for b64 in images_b64png:
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": parts},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        # Token parameter depends on model family
        if self.model.startswith("gpt-5"):
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens

        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=120,
        )

        if r.status_code != 200:
            # Print the actual OpenAI error message (this is vital!)
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


class ClaudeVisionClient(VisionLLMClient):
    def __init__(self, model: str, api_key: str, temperature: float = 0.0, max_tokens: int = 256):
        self.model = model
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        content: List[Dict[str, Any]] = [{"type": "text", "text": user}]
        for b in images_b64png:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b},
            })

        payload = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": max(64, self.max_tokens),
        }

        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        blocks = data.get("content", [])
        if not blocks:
            raise ValueError(f"Claude returned no content: {data}")
        text = ""
        for b in blocks:
            if b.get("type") == "text":
                text += b.get("text", "")
        return _extract_json(text)


class GeminiVisionClient(VisionLLMClient):
    def __init__(self, model: str, api_key: str, temperature: float = 0.0, max_tokens: int = 256):
        self.model = model
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        model = self.model
        if not model.startswith("models/"):
            model = "models/" + model
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={self.api_key}"

        # Gemini expects inline bytes; decode base64 -> bytes
        parts: List[Dict[str, Any]] = [{"text": system + "\n\n" + user}]
        for b64 in images_b64png:
            if isinstance(b64, (bytes, bytearray)):
                b64 = b64.decode("utf-8")
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": max(64, self.max_tokens),
                "responseMimeType": "application/json",
            },
        }

        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"Gemini returned no candidates: {data}")
        parts_out = candidates[0].get("content", {}).get("parts", [])
        if not parts_out:
            raise ValueError(f"Gemini returned no parts: {data}")
        text = parts_out[0].get("text", "")
        return _extract_json(text)


def load_llms(path: str) -> List[LLMConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: List[LLMConfig] = []
    for r in raw:
        out.append(LLMConfig(
            name=r["name"],
            provider=r["provider"],
            model=r["model"],
            api_key=_resolve_api_key(r.get("api_key", "")),
            base_url=r.get("base_url", "https://api.openai.com/v1"),
            temperature=float(r.get("temperature", 0.0)),
            max_tokens=int(r.get("max_tokens", 256)),
            supports_vision=bool(r.get("supports_vision", True)),
        ))
    return out


def make_client(cfg: LLMConfig) -> VisionLLMClient:
    p = (cfg.provider or "").lower().strip()
    if p in ("openai_compat", "openai"):
        return OpenAICompatVisionClient(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    if p in ("anthropic", "claude"):
        return ClaudeVisionClient(cfg.model, cfg.api_key, cfg.temperature, cfg.max_tokens)
    if p in ("gemini", "google"):
        return GeminiVisionClient(cfg.model, cfg.api_key, cfg.temperature, cfg.max_tokens)
    raise ValueError(f"Unknown provider: {cfg.provider}")
