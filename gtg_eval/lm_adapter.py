import time
import enum
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


class LiteLLM:
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
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

    async def completion(self, **kwargs):
        """Wrapper for litellm.completion that tracks token usage."""
        start_time = time.time()
        response = await litellm.acompletion(max_retries=5, **self._kwargs, **kwargs)
        elapsed = time.time() - start_time

        # Extract token usage
        prompt_tokens = 0
        completion_tokens = 0
        if usage := getattr(response, "usage", None):
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Update token tracker
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens
        self.last_total_tokens = total_tokens

        # Log token usage
        logger.info(
            "Completion finished",
            elapsed_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return response


class AdapterType(enum.StrEnum):
    LITELLM = "litellm"


_TYPE_TO_CLASS = {
    AdapterType.LITELLM: LiteLLM,
}


def make_lm_adapter(adapter_type: AdapterType, **kwargs) -> LMAdapter:
    if adapter_type not in _TYPE_TO_CLASS:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return _TYPE_TO_CLASS[adapter_type](**kwargs)
