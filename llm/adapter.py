from typing import Dict, List, Any, Optional, Iterator,Union
from abc import ABC, abstractmethod
from config.schema import LLMConfig

class LLMAdapter(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model
        self.timeout = config.timeout
        self.max_retries = config.max_retries

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        统一聊天接口。

        Returns (非流式):
            {
                "content": "Hello!",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "search", "arguments": "{\"q\": \"...\"}"}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "model": "gpt-4o"
            }

        Returns (流式): generator of delta chunks
        """
        pass

    @abstractmethod
    def _setup_client(self):
        """初始化底层客户端（如 OpenAI、Requests session）"""
        pass