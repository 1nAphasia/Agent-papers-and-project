from pydantic import BaseModel, Field
from typing import List, Optional
import json

class ToolCall(BaseModel):
    name: str
    args: dict

class LLMResponse(BaseModel):
    finish_reason: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    @classmethod
    def from_openai(cls, response: dict) -> "LLMResponse":
        """将 OpenAI 原始回复解析成统一结构"""
        choice = response["choices"][0]
        message = choice.get("message", {})
        return cls(
            finish_reason=choice.get("finish_reason"),
            content=message.get("content"),
            tool_calls=[
                ToolCall(
                    name=tc["function"]["name"],
                    args=json.loads(tc["function"]["arguments"]),
                )
                for tc in message.get("tool_calls", [])
            ]
            if message.get("tool_calls")
            else None,
        )
