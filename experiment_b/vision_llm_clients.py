import os
import json
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class LLMConfig:
    name: str
    provider: str  # openai | anthropic | gemini
    model: str
    api_key: str


def _resolve_api_key(k: str) -> str:
    if k.startswith("ENV:"):
        return os.environ.get(k.replace("ENV:", ""), "")
    return k


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        s = text.find("{")
        e = text.rfind("}")
        if s >= 0 and e > s:
            return json.loads(text[s:e+1])
        raise ValueError(f"LLM did not return JSON. Output:\n{text}")


class VisionLLMClient:
    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIVisionClient(VisionLLMClient):
    """
    Uses OpenAI Responses API via REST (no SDK dependency).
    """
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        content = [{"type": "input_text", "text": user}]
        for b in images_b64png:
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b}"})

        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()

        # Try to get unified output text
        text = data.get("output_text", "")
        if not text:
            # Fallback: scan output items
            parts = []
            for item in data.get("output", []) or []:
                for c in item.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text") and c.get("text"):
                        parts.append(c["text"])
            text = "\n".join(parts).strip()

        return _extract_json(text)


class ClaudeVisionClient(VisionLLMClient):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        content = [{"type": "text", "text": user}]
        for b in images_b64png:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b},
            })

        payload = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 700,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=90)
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
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def complete_json(self, *, system: str, user: str, images_b64png: List[str]) -> Dict[str, Any]:
        # Google Generative Language API (v1beta)
        model = self.model
        if not model.startswith("models/"):
            model = "models/" + model
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={self.api_key}"

        parts: List[Dict[str, Any]] = [{"text": system + "\n\n" + user}]
        for b in images_b64png:
            parts.append({
                "inline_data": {"mime_type": "image/png", "data": base64.b64decode(b)}
            })

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 700,
                # Request JSON if supported; still keep robust parsing
                "responseMimeType": "application/json",
            },
        }

        r = requests.post(url, json=payload, timeout=90)
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
    out = []
    for r in raw:
        out.append(LLMConfig(
            name=r["name"],
            provider=r["provider"],
            model=r["model"],
            api_key=_resolve_api_key(r["api_key"]),
        ))
    return out


def make_client(cfg: LLMConfig) -> VisionLLMClient:
    p = cfg.provider.lower().strip()
    if p == "openai":
        return OpenAIVisionClient(cfg.model, cfg.api_key)
    if p in ("anthropic", "claude"):
        return ClaudeVisionClient(cfg.model, cfg.api_key)
    if p in ("gemini", "google"):
        return GeminiVisionClient(cfg.model, cfg.api_key)
    raise ValueError(f"Unknown provider: {cfg.provider}")
