from __future__ import annotations
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from config.settings import get_settings


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
        )

    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
        )

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )