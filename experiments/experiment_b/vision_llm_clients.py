import os
import json
import time
import random
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


def _post(url, *, headers=None, json=None, timeout=120, max_retries=6):
    """POST with retry + backoff on rate-limit / transient overload (429, 5xx, 529).
    Returns the final Response (caller checks status_code)."""
    r = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.request("POST", url, headers=headers, json=json, timeout=timeout)
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))
            continue
        if r.status_code in (429, 500, 502, 503, 529) and attempt < max_retries:
            ra = (r.headers or {}).get("retry-after", "")
            wait = float(ra) if ra.replace(".", "", 1).isdigit() else min(30.0, 2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait)
            continue
        return r
    return r


@dataclass
class LLMConfig:
    """
    Provider-agnostic LLM configuration.

    Expected JSON fields:
      - name: human-readable name used for filenames / tables
      - provider: "openai_compat" | "openai" | "anthropic" | "gemini"
      - model: provider model id
      - api_key: literal key OR "ENV:VARNAME"
      - base_url: (openai/openai_compat only) e.g. "https://api.openai.com/v1" or "http://localhost:8000/v1"
      - temperature: float (default 0.0)
      - max_tokens: int (default 256)
      - supports_vision: bool (default True). If False, runner skips this model for image inputs.
    """
    name: str
    provider: str
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_tokens: int = 256
    supports_vision: bool = True
    reasoning: bool = False   # o-series / reasoning models: skip temperature, use max_completion_tokens
    service_tier: Optional[str] = None   # OpenAI only, e.g. "flex" (~50% off, best-effort latency)


def _resolve_api_key(k: str) -> str:
    if (k or "").startswith("ENV:"):
        k = os.environ.get(k.replace("ENV:", ""), "")
    # Strip whitespace and stray surrounding quotes (incl. smart quotes) that sneak in
    # when a key is exported like  export X="key"  and the quotes become curly on paste.
    return (k or "").strip().strip("\"'“”‘’").strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """Accept exact JSON or a response containing a JSON object (handles markdown fences)."""
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
    OpenAI Chat Completions-compatible client (works for OpenAI and OpenAI-compatible servers,
    e.g. vLLM, LM Studio, Together AI).

    Uses /chat/completions with image_url content blocks for vision.
    response_format=json_object is only sent to openai.com (not all servers support it).
    """
    def __init__(self, model: str, api_key: str, base_url: str, temperature: float = 0.0,
                 max_tokens: int = 256, reasoning: bool = False, service_tier: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.reasoning = bool(reasoning)
        self.service_tier = service_tier or None
        self._is_openai = "openai.com" in self.base_url
        m = model.lower()
        self._o_series = m.startswith(("o1", "o3", "o4"))           # reasoning models
        self._use_max_completion = self._o_series or m.startswith("gpt-5")

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        parts = [{"type": "text", "text": user}]
        for b64 in images_b64png:
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": parts}]

        def build(send_temperature: bool, send_service_tier: bool) -> Dict[str, Any]:
            p: Dict[str, Any] = {"model": self.model, "messages": messages}
            if self._is_openai:
                p["response_format"] = {"type": "json_object"}
            # gpt-5* and o-series use max_completion_tokens; everyone else uses max_tokens.
            p["max_completion_tokens" if self._use_max_completion else "max_tokens"] = self.max_tokens
            # gpt-5* "think" by default and would eat the token budget; as a STANDARD (non-reasoning)
            # model we keep effort minimal. (o-series / reasoning models keep their default effort.)
            if self._is_openai and self.model.lower().startswith("gpt-5") and not self.reasoning:
                p["reasoning_effort"] = "none"
            # Flex / priority processing (OpenAI only): ~50% cheaper best-effort tier.
            if self._is_openai and self.service_tier and send_service_tier:
                p["service_tier"] = self.service_tier
            if send_temperature:
                p["temperature"] = self.temperature
            return p

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # Reasoning / o-series models reject a non-default temperature, so don't send one.
        send_temp = not (self._o_series or self.reasoning)
        send_tier = True
        # Flex requests are best-effort and can queue longer, so allow more wall-clock.
        timeout = 600 if self.service_tier == "flex" else 180
        r = _post(url, headers=headers, json=build(send_temp, send_tier), timeout=timeout)
        # Auto-fallback: drop an unsupported param the API complained about, then retry.
        if r.status_code == 400 and send_tier and "service_tier" in r.text.lower():
            send_tier = False
            r = _post(url, headers=headers, json=build(send_temp, send_tier), timeout=180)
        if r.status_code == 400 and send_temp and "temperature" in r.text.lower():
            send_temp = False
            r = _post(url, headers=headers, json=build(send_temp, send_tier), timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")

        content = r.json()["choices"][0]["message"]["content"]
        return _extract_json(content)


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

        def build(send_temperature: bool) -> Dict[str, Any]:
            p = {
                "model": self.model,
                "system": system,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max(64, self.max_tokens),
            }
            if send_temperature:
                p["temperature"] = self.temperature
            return p

        r = _post(url, headers=headers, json=build(True), timeout=120)
        # Newer Claude models (e.g. Sonnet 5) deprecate temperature -> retry without it.
        if r.status_code == 400 and "temperature" in r.text.lower():
            r = _post(url, headers=headers, json=build(False), timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()

        blocks = data.get("content", [])
        if not blocks:
            raise ValueError(f"Claude returned no content: {data}")
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
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

        parts: List[Dict[str, Any]] = [{"text": system + "\n\n" + user}]
        for b64 in images_b64png:
            if isinstance(b64, (bytes, bytearray)):
                b64 = b64.decode("utf-8")
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})

        gen_cfg = {
            "temperature": self.temperature,
            # Gemini 3.x flash keeps "thinking" even with thinkingBudget=0 (the flag is ignored
            # or unsupported), and thinking tokens count against maxOutputTokens -> the visible
            # JSON gets truncated. Give a large budget so thinking + the short JSON both fit.
            "maxOutputTokens": max(8192, self.max_tokens),
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        }
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen_cfg}

        r = _post(url, json=payload, timeout=120)
        # Not all model versions accept thinkingConfig; retry once without it if that's the complaint.
        if r.status_code == 400 and "thinking" in r.text.lower():
            gen_cfg.pop("thinkingConfig", None)
            gen_cfg["maxOutputTokens"] = max(2048, self.max_tokens)   # give thinking room instead
            r = _post(url, json=payload, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"Gemini returned no candidates: {data}")
        parts_out = candidates[0].get("content", {}).get("parts", [])
        if not parts_out:
            raise ValueError(f"Gemini returned no parts: {data}")
        return _extract_json(parts_out[0].get("text", ""))


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
            reasoning=bool(r.get("reasoning", False)),
            service_tier=r.get("service_tier") or None,
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
            reasoning=cfg.reasoning,
            service_tier=cfg.service_tier,
        )
    if p in ("anthropic", "claude"):
        return ClaudeVisionClient(cfg.model, cfg.api_key, cfg.temperature, cfg.max_tokens)
    if p in ("gemini", "google"):
        return GeminiVisionClient(cfg.model, cfg.api_key, cfg.temperature, cfg.max_tokens)
    raise ValueError(f"Unknown provider: {cfg.provider}")
