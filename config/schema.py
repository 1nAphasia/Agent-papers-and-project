from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-reasoner"
    api_key: Optional[str] = None  # 从环境变量读取
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60
    max_retries: int = 3

class AgentConfig(BaseModel):
    role: str
    max_steps: int = 5
    enable_rag: bool = False
    tool_timeout: int = 30

class ToolConfig(BaseModel):
    enabled: List[str] = ["http_get"]
    sandbox_enabled: bool = True
    network_access: bool = False

# class MemoryConfig(BaseModel):
#     vector_store: str = "chroma"
#     embedding_model: str = "text-embedding-3-small"

class AppConfig(BaseModel):
    name: str = "MultiAgentFramework"
    env: str = "development"
    debug: bool = True

class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_format: bool = True

class GlobalConfig(BaseModel):
    app: AppConfig = AppConfig()
    llm: LLMConfig = LLMConfig()
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    tools: ToolConfig = ToolConfig()
    # memory: MemoryConfig = MemoryConfig()
    logging: LoggingConfig = LoggingConfig()