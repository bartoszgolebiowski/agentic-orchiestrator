from __future__ import annotations

from engine.llm.completion import get_model, get_raw_client
from engine.llm.entities import LLMClientEntity


class LLMClientFactory:
    @staticmethod
    def create() -> LLMClientEntity:
        return LLMClientEntity(client=get_raw_client(), default_model=get_model())
