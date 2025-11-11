import requests
from typing import Dict, List, Any,Optional
from openai import OpenAI
import re
import json
from termcolor import colored
import os
import sys
import time
import multiprocessing
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
# from hipporag import HippoRAG
from copy import deepcopy
from abc import ABC,abstractmethod
from llm.adapter import LLMAdapter
from memory.base import SimpleMemory



current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

# from configs import CommonConfig
from config.schema import LLMConfig
# from llm_agent.context import BaseContextManager
# from llm_agent.tools.tool_manager import BaseToolManager

class BaseAgent(ABC):
    def __init__(self,name:str,role:str='',config:Optional[Dict]=None):
        self.name=name
        self.role=role
        self.config=config or {}
        self._initialized=False
        self.memory=SimpleMemory()

    def initialize(self) -> None:
        """可选：延迟初始化资源（如 LLM 客户端、记忆模块）"""
        if not self._initialized:
            self._setup()
            self._initialized = True

    @abstractmethod
    def _setup(self) -> None:
        """子类实现具体初始化逻辑（如加载 LLM、连接记忆库）"""
        raise NotImplementedError("未实现_setup方法")


    @abstractmethod
    def act(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent 的核心行为方法。
        Orchestrator 调用此方法触发 Agent 执行。

        Args:
            input_data: 包含上下文、消息历史、任务描述等。
                示例：
                {
                    "task_id": "t123",
                    "messages": [{"role": "user", "content": "..."}],
                    "available_tools": ["search", "calc"],
                    "memory_context": "...",
                }

        Returns:
            结构化响应，必须包含以下之一：
            - "response": 纯文本回复（对话结束）
            - "tool_calls": 工具调用列表（需 Orchestrator 执行后回填）
            - "next_agents": 下一步应激活的 Agent 列表（用于协作）

            示例返回：
            {
                "response": None,
                "tool_calls": [
                    {"tool": "http_get", "args": {"url": "https://api.example.com"}, "call_id": "c1"}
                ],
                "metadata": {"reasoning": "需要获取外部数据..."}
            }
        """
        raise NotImplementedError("未实现act方法")

    def observe(self, event: Dict[str, Any]) -> None:
        """
        接收系统事件（如工具执行结果、其他 Agent 消息）。
        默认实现：可被子类覆盖以实现反应式行为。
        """
        raise NotImplementedError("未实现observe方法")
    
    def remember(self,key:str,value:Any)->None:
        self.memory.add(key,value)
        
    def recall(self, key: str) -> Optional[Any]:
        return self.memory.get(key)

    def shutdown(self) -> None:
        """释放资源（如关闭连接、保存状态）"""
        raise NotImplementedError("未实现shutdown方法")



class SimpleAgent(BaseAgent):
    def __init__(self, name, role = '', config = None):
        super().__init__(name, role, config)
        self.llm:Optional[LLMAdapter]=None

    def _setup(self):
        self.llm=LLMAdapter(self.config)

    def act(self,input_data:Dict[str,Any])->Dict[str,Any]:
        messages=input_data.get("messages",[])
        available_tools=input_data.get('available_tools',[])
        response=self.llm.chat(
            messages=messages,
            tools=available_tools,
            tool_choice='auto'
            )
        if response.get("tool_calls"):
            return {
                "response": None,
                "tool_calls": [
                    {
                        "tool": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"]),
                        "call_id": tc["id"]
                    }
                    for tc in response["tool_calls"]
                ],
                "metadata": {}
            }
        else:
            return {
                "response": response["content"],
                "tool_calls": None,
                "metadata": {}
            }
    
    def observe(self, event):
        if event.get('type')=='tool_result':
            pass
    
    def shutdown(self):
        return super().shutdown()
    

if __name__ =="__main__":
    from multiprocessing import freeze_support
    freeze_support()

    cfg={
        'model':'deepseek-reasoner',
        'base_url':r'https://api.deepseek.com',
        'api_key':'sk-716f93e2299940b58b5838939cade9d0',
        'generation_config': {
            'max_tokens':5000,
            'temperature':0.5
            },
        'stop_condition': r'<code[^>]*>((?:(?!<code).)*?)</code>',
        'tool_condition': r'<code[^>]*>((?:(?!<code).)*?)</code>',
        'is_debug':True
        }
    llm_config=LLMConfig(cfg)
    base_agent=SimpleAgent(cfg)
    
    base_agent.step("浓硫酸的化学性质有哪些？")