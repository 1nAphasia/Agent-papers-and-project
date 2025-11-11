from typing import Dict,Any,Optional,List,Tuple
from agent.base import BaseAgent
import logging
from uuid import uuid4
from tools.mcp_client import MCPClient
from agent.response import LLMResponse
import asyncio
import time
import json

logger=logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.agents: Dict[str,BaseAgent]={}
        self.tool_list:List[Dict]=[]
        self.task_contexts:Dict[str,dict]={}
        self.mcp_clients:Dict[str,Dict]={}

    def register_agent(self,name:str,agent:BaseAgent):
        if name in self.agents:
            raise ValueError(f"Agent {name} already registered")
        self.agents[name]=agent
        agent.initialize()
    
    def unregister_agent(self,name:str):
        self.agents[name].shutdown()
        del self.agents[name]
        logger.info(f'registered agent:{name}')


    def register_mcp_client(self,name:str,client:MCPClient):
        self.mcp_clients[name]=client
        self._update_tool_schemas()

    
    def dispatch(self,
                 agent_name:str,
                 input_data:Dict[str,Any],
                 task_id:Optional[str]=None
                 )->Dict[str,Any]:
        agent=self.agents[agent_name]
        input_data['available_tools']=[{"name":name,"description":tool.discription,"input_schema":tool.input_schema} 
                                       for name,tool in self.tools.items()]
        return agent.act(input_data)
    
    def start_task(self,initial_agent:str,user_input:str)->str:
        task_id=str(uuid4())

        #用于管理任务上下文的task_context。messages应包括用户的对话和回复。
        self.task_contexts[task_id] = {
            "messages": [ {"role":"user","content":user_input}],
            "active_agent": "weather_agent",
            "step_count": 0,
            "max_steps": 10,
            "status": "running",  # or "finished","errored"
            "request_id": task_id,
            "last_response": None
        }
        logger.info(f"开始任务,任务id为： {task_id} 。初始Agent： {initial_agent}")
        return task_id
    
    
    async def step(self, task_id: str) -> Tuple[bool, Optional[str]]:
        if task_id not in self.task_contexts:
            raise ValueError(f"task_id为{task_id} 的任务不存在。")
        ctx = self.task_contexts[task_id]

        if ctx.get("status") != "running":
            return False, f"任务状态 {ctx.get('status')}"

        agent_name = ctx["active_agent"]
        agent = self.agents[agent_name]

        # 为agent构建可用的工具明细。
        available_tools = []
        for item in self.tool_list:
            client = item["client"]
            t = item["tool"]["function"]
            available_tools.append({
                "client": client,
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            })

        input_data = {
            "task_id": task_id,
            "messages": ctx["messages"],
            "available_tools": available_tools,
            "request_id": ctx.get("request_id")
        }

        # 异步调用agent
        try:
            response: LLMResponse = await agent.act(input_data)  # agent.act must be async
        except Exception as e:
            logger.exception("agent.act 调用失败: %s", e)
            ctx["status"] = "errored"
            return False, f"agent failure: {e}"

        ctx["step_count"] = ctx.get("step_count") + 1
        ctx["last_response"] = response

        # 处理agent单步结果
        if response.finish_reason == "content":
            # 加入LLM返回结果,结束任务
            assistant_msg = {"role":"assistant","content": response.content}
            ctx["messages"].append(assistant_msg)
            ctx["status"] = "finished"
            return True, response.content

        if response.finish_reason == "tool_calls":
            # response.tool_calls: list of ToolCall objects (name,args,...)
            # 验证输入并逐步调用工具
            tool_results = []
            for tc in response.tool_calls:
                # 找到工具调用中和tool_list中的tool的信息
                matching = [t for t in self.tool_list if t["tool"]["function"]["name"] == tc.name]
                if not matching:
                    tool_results.append({"name": tc.name, "ok": False, "error": "unknown tool"})
                    continue

                tool_entry = matching[0]
                client_name = tool_entry["client"]
                client_obj = self.mcp_clients[client_name]

                # 一个根据定义验证输入参数的伪实现
                # validate(tc.args, tool_entry["tool"]["function"]["parameters"])

                # 调用工具(带有超时检查和错误检查)
                try:
                    timeout = tc.get("timeout", 15)
                    start = time.time()
                    res = await asyncio.wait_for(client_obj.call_tool(tc.name, tc.args), timeout=timeout)
                    duration = time.time() - start
                    tool_results.append({"name": tc.name, "ok": True, "result": res, "duration": duration})
                except asyncio.TimeoutError:
                    tool_results.append({"name": tc.name, "ok": False, "error": "timeout"})
                except Exception as e:
                    tool_results.append({"name": tc.name, "ok": False, "error": str(e)})

            # 将工具结果加入messages
            for tr in tool_results:
                content = json.dumps(tr, ensure_ascii=False)
                ctx["messages"].append({"role":"tool", "content": content, "name": tr.get("name")})
            ctx.setdefault("tool_history", []).extend(tool_results)

            # 向agent发送事件(允许agent将工具调用结果写进自己的记忆中)
            try:
                agent.observe({"type":"tool_results", "results": tool_results, "task_id": task_id})
            except Exception:
                logger.exception("agent.observe failed")

            # 决定是继续还是回到agent System中
            # 这里选择继续,但是由max_step限制最大步数。
            if ctx["step_count"] >= ctx.get("max_steps", 10):
                ctx["status"] = "errored"
                return False, "超过最大调用步数。"
            # recursively call step OR let caller call step again; here we choose to call agent.act again now:
            return await self.step(task_id)
        # 未知的结束原因(？)
        ctx["status"] = "errored"
        return False, "unsupported finish reason"
        

    def _update_tool_schemas(self) -> List[Dict]:
        """
        获取所有mcp服务器的工具,并集成到列表中。
        """
        self.tool_list= [{
            "client": name,
            "tool":
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
        }
            for name,client in self.mcp_clients.items()
            for tool in client.list_tools()
        ]
    
    async def execute_tools_and_continue(
            self,
            task_id:str,
            tool_calls:List[Dict],
            current_agent:str,
    )->Dict[str,Any]:
        results=[]
        for tc in tool_calls:
            tool_name=tc.name
            for tool in self.tool_list:
                if tool_name==tool["tool"]["function"]["name"]:
                    result=await self.mcp_clients[tool["client"]].call_tools(tc.name,tc.args)
                    results.append(result)
        
        if not results:
            raise ValueError("目标工具和调用不一致,请检查调用")
        
        # event={
        #     "type":"tool_results",
        #     "results":results,
        #     "task_id":task_id
        # }
        # next_agent=current_agent # 简化：默认回原 Agent

        return {"role":"tools","content":results}
        
    def shutdown(self):
        for agent in self.agents.values():
            agent.shutdown()
        for client in self.mcp_clients.values():
            client.shutdown()
