import asyncio
import json
import time
from agent.base import BaseAgent
from llm.factory import get_llm_adapter
from config.prompts.loader import load_prompt_template
from config.logger import get_logger
from agent.response import LLMResponse
import contextvars
from typing import Optional,Dict,List,Any

logger=get_logger(__name__)

request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)

def get_request_id() -> Optional[str]:
    return request_id_var.get()

class AsyncAgent(BaseAgent):
    def __init__(self, name, role = '', config = None):
        super().__init__(name, role, config)

    def initialize(self):
        return super().initialize()
    
    def _setup(self):
        self.llm=get_llm_adapter(self.config)
        self.prompt_template=load_prompt_template(self.role)
    
    async def act(self, input_data:Dict[str,Any]):
        logger.info(self.role+"开始运行。任务为："+input_data.get('task',''))

        # 解析输入并设置 request_id
        task_id = input_data.get("task_id")
        messages = input_data.get("messages", [])

        tool_results=[message.content for message in messages if message["role"] == "tool"]

        available_tools = input_data.get("available_tools", [])
        rid = input_data.get("request_id")

        # 把 request_id 绑定到当前协程上下文
        token = request_id_var.set(rid)

        logger.info(f"[rid={get_request_id()}] Agent {self.name} Act 一次,task_id 为 {task_id}")
        # 获取当前task_id,并根据id获取由自己管理的对话历史。
        ctx={
            'available_tools':available_tools,
            'history':messages[:-1],
            'current_user':messages[0],
            'tool_results':tool_results if tool_results else None
        }
        input_messages=self.prompt_template.render(ctx)
        # 调用 LLM（将同步调用包装为线程）
        start_ts = time.time()
        try:
            if asyncio.iscoroutinefunction(self.llm.chat):
                # 异步的话,直接用
                raw_response = await self.llm.chat(
                    messages=input_messages,
                    tools=available_tools,
                    tool_choice="auto",
                    stream=False
                )
            else:
                # 同步的话,用线程包装
                raw_response = await asyncio.to_thread(
                    self.llm.chat,
                    messages,
                    available_tools,
                    "auto",
                    False
                )
        except Exception as e:
            logger.exception(f"[rid={get_request_id()}] LLM调用失败: {e}")
            # 恢复 contextvar
            request_id_var.reset(token)
            # 可以返回一个含 error 的 LLMResponse 或抛出异常让 manager 处理
            raise

        latency = time.time() - start_ts
        logger.info("[rid=%s] 调用LLM用时 %.3fs", get_request_id(), latency)

        # 将 raw_response 转为统一的 LLMResponse
        resp_obj = LLMResponse.from_openai(raw_response.model_dump())

        # agent 可在此阶段选择把部分内容写入自己的 memory
        # 例如把用户消息和 assistant 输出记进 agent 内部短期历史：
        try:
            # 记忆 key 可结合 task_id, 保证 per-task per-agent 隔离
            hist_key = f"history_{task_id}"
            history = self.recall(hist_key) or []
            history.extend([
                {"role": "user", "content": messages[-1]["content"] if messages else ""},
                {"role": "assistant", "content": resp_obj.content or ""}
            ])
            self.remember(hist_key, history)
        except Exception:
            logger.exception(f"[rid= {get_request_id()}] memory write failed")

        # 清理 contextvar 并返回 LLMResponse
        request_id_var.reset(token)
        return resp_obj
        
    def observe(self, event):
        """
        得到工具执行结果。
            event={
                "type":"tool_results",
                "results":results,
                "task_id":task_id
            }
        """
        logger.info(f"得到工具执行结果：{event["results"]}。继续运行。")
        task_id = event["task_id"] 
        self.remember(f"tool_results_{task_id}", event["results"])

        
    def shutdown(self):
        return 