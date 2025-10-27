
import asyncio
from typing import Dict, Any, Optional, List, Callable
from llm.factory import get_llm_adapter
from agent.base import BaseAgent
from tools.base import BaseTool
from tools.toolrunner import ToolRunner
import logging
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Orchestrator for agents and tools.
    - 异步 dispatch 接口
    - 事件总线（简单实现）
    - tool 调用前后钩子（用于审计/metrics/安全）
    """
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.agents: Dict[str, BaseAgent] = {}
        self.tools: Dict[str, BaseTool] = {}
        self.tool_runner = ToolRunner()
        self.active_sessions: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self.loop = loop or asyncio.get_event_loop()

        # hooks / middleware: callables with signature (ctx:dict)
        self.pre_dispatch_hooks: List[Callable[[Dict[str, Any]], None]] = []
        self.post_dispatch_hooks: List[Callable[[Dict[str, Any]], None]] = []
        self.pre_tool_hooks: List[Callable[[Dict[str, Any]], None]] = []
        self.post_tool_hooks: List[Callable[[Dict[str, Any]], None]] = []

        # internal queue for tasks when running locally
        self.task_queue: asyncio.Queue = asyncio.Queue()

        # background worker
        self._worker_task = self.loop.create_task(self._task_worker())

    async def _task_worker(self):
        while True:
            item = await self.task_queue.get()
            if item is None:
                break
            try:
                await self._handle_dispatch(**item)
            except Exception:
                logger.exception("Error handling queued task")
            finally:
                self.task_queue.task_done()

    async def register_agent(self, name: str, agent: BaseAgent):
        async with self._lock:
            if name in self.agents:
                raise ValueError(f"Agent {name} already registered")
            self.agents[name] = agent
        # lifecycle init can be async or sync
        init_fn = getattr(agent, "initialize", None)
        if asyncio.iscoroutinefunction(init_fn):
            await init_fn()
        elif callable(init_fn):
            init_fn()
        logger.info(f"Registered agent: {name}")

    async def unregister_agent(self, name: str):
        async with self._lock:
            agent = self.agents.pop(name, None)
        if not agent:
            return
        shutdown_fn = getattr(agent, "shutdown", None)
        if asyncio.iscoroutinefunction(shutdown_fn):
            await shutdown_fn()
        elif callable(shutdown_fn):
            shutdown_fn()
        logger.info(f"Unregistered agent: {name}")

    async def register_tool(self, name: str, tool: BaseTool):
        async with self._lock:
            self.tools[name] = tool
        logger.info(f"Registered tool: {name}")

    async def dispatch(self,
                       agent_name: str,
                       input_data: Dict[str, Any],
                       task_id: Optional[str] = None,
                       enqueue: bool = False) -> Dict[str, Any]:
        """
        Public dispatch entry. If enqueue True, put task to internal queue for async processing.
        Returns final result when not enqueued, otherwise returns task metadata.
        """
        ctx = {
            "task_id": task_id or str(uuid4()),
            "agent": agent_name,
            "input": input_data,
            "ts": datetime.utcnow().isoformat()
        }
        for hook in self.pre_dispatch_hooks:
            try:
                hook(ctx)
            except Exception:
                logger.exception("pre_dispatch hook error")

        if enqueue:
            await self.task_queue.put({"agent_name": agent_name, "input_data": input_data, "task_id": ctx["task_id"]})
            return {"task_id": ctx["task_id"], "status": "queued"}

        return await self._handle_dispatch(agent_name=agent_name, input_data=input_data, task_id=ctx["task_id"])

    async def _handle_dispatch(self, agent_name: str, input_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        # validate agent
        if agent_name not in self.agents:
            return {"error": f"agent {agent_name} not registered", "task_id": task_id}

        agent = self.agents[agent_name]

        # attach available tools metadata (safe subset)
        input_data = dict(input_data)  # copy
        input_data['available_tools'] = [
            {"name": n, "description": t.description, "input_schema": getattr(t, "input_schema", None)}
            for n, t in self.tools.items()
        ]

        # call agent.act (support async/sync)
        act_fn = getattr(agent, "act", None)
        if act_fn is None:
            return {"error": "agent has no act method", "task_id": task_id}

        try:
            if asyncio.iscoroutinefunction(act_fn):
                action = await act_fn(input_data)
            else:
                action = act_fn(input_data)
        except Exception:
            logger.exception("agent.act failed")
            return {"error": "agent execution error", "task_id": task_id}

        # post dispatch hook
        post_ctx = {"task_id": task_id, "agent": agent_name, "action": action}
        for hook in self.post_dispatch_hooks:
            try:
                hook(post_ctx)
            except Exception:
                logger.exception("post_dispatch hook error")

        # handle tool calls if present
        tool_calls = action.get("tool_calls") if isinstance(action, dict) else None
        if tool_calls:
            return await self.execute_tools_and_continue(task_id, tool_calls, current_agent=agent_name, origin_action=action)
        # otherwise return action/result
        return {"task_id": task_id, "result": action}

    async def execute_tools_and_continue(self,
                                         task_id: str,
                                         tool_calls: List[Dict[str, Any]],
                                         current_agent: str,
                                         origin_action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        批量顺序执行 tool_calls（可改为并行执行），在每次回填 observe 给指定 agent。
        每个 tool_call 结构应为: {"tool": "name", "args": {...}, "call_id": optional}
        """
        results = []
        for tc in tool_calls:
            tool_name = tc.get("tool")
            call_id = tc.get("call_id", str(uuid4()))
            ctx = {
                "task_id": task_id,
                "call_id": call_id,
                "agent": current_agent,
                "tool": tool_name,
                "args": tc.get("args", {}),
                "ts": datetime.utcnow().isoformat()
            }

            # pre-tool hooks (audit/permission/schema checks)
            for hook in self.pre_tool_hooks:
                try:
                    hook(ctx)
                except Exception:
                    logger.exception("pre_tool hook error")

            if tool_name not in self.tools:
                res = {"error": f"Tool {tool_name} not found", "call_id": call_id}
            else:
                # Use tool_runner to execute with context and safe params
                try:
                    res = await self.tool_runner.run_async(
                        tool=self.tools[tool_name],
                        params=ctx["args"],
                        context={"task_id": task_id, "agent": current_agent, "call_id": call_id}
                    )
                except Exception:
                    logger.exception("ToolRunner error")
                    res = {"error": "tool execution failed", "call_id": call_id}

            results.append(res)

            # post-tool hooks (audit/metrics)
            post_ctx = {"task_id": task_id, "call_id": call_id, "tool": tool_name, "result": res}
            for hook in self.post_tool_hooks:
                try:
                    hook(post_ctx)
                except Exception:
                    logger.exception("post_tool hook error")

            # 回填给 agent（observe）
            evt = {"type": "tool_result", "task_id": task_id, "call_id": call_id, "tool": tool_name, "result": res}
            obs_fn = getattr(self.agents[current_agent], "observe", None)
            try:
                if obs_fn:
                    if asyncio.iscoroutinefunction(obs_fn):
                        await obs_fn(evt)
                    else:
                        obs_fn(evt)
            except Exception:
                logger.exception("agent.observe failed")

        # 默认回到原 agent；可以在 action 中指定 next_agent
        return {"task_id": task_id, "results": results, "next_action": "continue", "next_agent": current_agent}

    async def observe_event(self, agent_name: str, event: Dict[str, Any]):
        """外部事件注入（可用于 webhooks / tool callbacks）"""
        if agent_name in self.agents:
            obs_fn = getattr(self.agents[agent_name], "observe", None)
            if asyncio.iscoroutinefunction(obs_fn):
                await obs_fn(event)
            elif callable(obs_fn):
                obs_fn(event)

    async def shutdown(self):
        # stop worker
        await self.task_queue.put(None)
        await self._worker_task
        # shutdown agents
        for name, agent in list(self.agents.items()):
            try:
                shutdown_fn = getattr(agent, "shutdown", None)
                if asyncio.iscoroutinefunction(shutdown_fn):
                    await shutdown_fn()
                elif callable(shutdown_fn):
                    shutdown_fn()
            except Exception:
                logger.exception("agent shutdown error")
        logger.info("AgentManager shutdown complete")
# ...existing code...