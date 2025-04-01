import enum
import os
import time
from typing import Any, Protocol

import litellm
from loguru import logger

# Gemini models do no support seed=.
litellm.drop_params = True


class LMAdapter(Protocol):
    last_prompt_tokens: int
    last_completion_tokens: int
    last_total_tokens: int

    def __init__(self, **kwargs): ...
    async def completion(self, **kwargs) -> Any: ...


class LastTokenUsageMixin:
    def _update_token_usage(self, response: Any):
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.last_prompt_tokens = getattr(usage, "prompt_tokens", 0)
        self.last_completion_tokens = getattr(usage, "completion_tokens", 0)
        self.last_total_tokens = self.last_prompt_tokens + self.last_completion_tokens


class LiteLLM(LastTokenUsageMixin):
    def __init__(self, **kwargs):
        # if not litellm.supports_vision(model=kwargs["model"]):
        #     logger.error("Model does not support vision", model=kwargs["model"])
        #     return 1
        # Add Gemini-specific safety settings if needed
        if "gemini" in kwargs.get("model", "").lower():
            kwargs["safety_settings"] = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

        self._kwargs = kwargs

    async def completion(self, **kwargs):
        """Wrapper for litellm.completion that tracks token usage."""
        start_time = time.time()
        response = await litellm.acompletion(max_retries=5, **self._kwargs, **kwargs)
        elapsed = time.time() - start_time
        self._update_token_usage(response)
        logger.info(
            "Completion finished",
            elapsed_seconds=elapsed,
            prompt_tokens=self.last_prompt_tokens,
            completion_tokens=self.last_completion_tokens,
            total_tokens=self.last_total_tokens,
        )
        return response


class VolcEngineBatch(LastTokenUsageMixin):
    def __init__(self, **kwargs):
        from volcenginesdkarkruntime import AsyncArk

        kwargs.pop("seed", None)
        self._kwargs = kwargs
        self.client = AsyncArk(
            api_key=os.environ.get("ARK_API_KEY"), timeout=60 * 60 * 3
        )

    async def completion(self, **kwargs):
        start_time = time.time()
        kwargs.pop("seed", None)
        response = await self.client.batch_chat.completions.create(
            **self._kwargs, **kwargs
        )
        elapsed = time.time() - start_time
        self._update_token_usage(response)
        logger.info(
            "Completion finished",
            elapsed_seconds=elapsed,
            prompt_tokens=self.last_prompt_tokens,
            completion_tokens=self.last_completion_tokens,
            total_tokens=self.last_total_tokens,
        )
        return response


class AdapterType(enum.StrEnum):
    LITELLM = "litellm"
    VOLCENGINE_BATCH = "volcengine_batch"


_TYPE_TO_CLASS = {
    AdapterType.LITELLM: LiteLLM,
    AdapterType.VOLCENGINE_BATCH: VolcEngineBatch,
}


def make_lm_adapter(adapter_type: AdapterType, **kwargs) -> LMAdapter:
    if adapter_type not in _TYPE_TO_CLASS:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return _TYPE_TO_CLASS[adapter_type](**kwargs)
