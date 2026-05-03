from __future__ import annotations
import os
from functools import lru_cache
from dotenv import load_dotenv
 
load_dotenv()
 
 
class Settings:
    # LLM
    llm_provider: str       = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str          = os.getenv("LLM_MODEL", "gpt-4o-mini")
    openai_api_key: str     = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str  = os.getenv("ANTHROPIC_API_KEY", "")
    google_api_key: str     = os.getenv("GOOGLE_API_KEY", "")
    groq_api_key: str       = os.getenv("GROQ_API_KEY", "")
 
    # Tools
    tavily_api_key: str     = os.getenv("TAVILY_API_KEY", "")
 
    # Memory
    redis_url: str          = os.getenv("REDIS_URL", "")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
 
    # Graph limits
    max_search_results: int  = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    max_critic_retries: int  = int(os.getenv("MAX_CRITIC_RETRIES", "2"))
 
    # Logging
    log_level: str          = os.getenv("LOG_LEVEL", "INFO")
 
    def validate(self) -> None:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if self.llm_provider == "google" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=google")
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required for web search")
 
 
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.validate()
    return s