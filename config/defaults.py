from typing import Dict, Any

DEFAULT_CONFIG = {
    "app": {
        "name": "MultiAgentFramework",
        "env": "development",
        "debug": True,
    },
    "llm": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-reasoner",
        "temperature": 0.7,
        "max_tokens": 1024,
        "timeout": 60,
        "max_retries": 3,
        "api_key": "sk-1798cb940eb54fafab8d6da24933e5db"
    },
    "agents": {
        "planner": {
            "role": "planner",
            "max_steps": 5,
            "enable_rag": False
        },
        "executor": {
            "role": "excutor",
            "tool_timeout": 30
        },
        "weather":{
            "role":"weather",
        },
        "faissmanager":{
            "role":"faissmanager",
            
        }
    },
    "tools": {
        "enabled": ["http_get", "calculator"],
        "sandbox": {
            "enabled": True,
            "network_access": False
        }
    },
    "memory": {
        "vector_store": "chroma",
        "embedding_model": "text-embedding-3-small"
    },
    "logging": {
        "level": "INFO",
        "json_format": True
    }
}