from llm.factory import get_llm_adapter
from typing import Dict,Any,Optional,List,Tuple
from agent.base import BaseAgent
from tools.base import BaseTool
from tools.toolrunner import ToolRunner
import logging
from uuid import uuid4

logger=logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.agents: Dict[str,BaseAgent]={}
        self.tools:Dict[str,BaseTool]={}
        self.tool_runner=ToolRunner()
        self.task_contexts:Dict[str,dict]={}

    def register_agent(self,name:str,agent:BaseAgent):
        if name in self.agents:
            raise ValueError(f"Agent {name} already registered")
        self.agents[name]=agent
        agent.initialize()
    
    def unregister_agent(self,name:str):
        self.agents[name].shutdown()
        del self.agents[name]
        logger.info(f'registered agent:{name}')


    def register_tool(self,name:str,tool:BaseTool):
        self.tools[name]=tool

    
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
        self.task_contexts[task_id]={
            "messages":[{"role":"user","content":user_input}],
            "active_agent":initial_agent,
            "step_count":0,
            "max_steps":10
        }
        logger.info(f"Started task {task_id} with agent {initial_agent}")
        return task_id
    
    def step(self,task_id:str)-> Tuple[bool,Optional[str]]:
        if task_id not in self.task_contexts:
            raise ValueError(f"Task {task_id} not found")
        context=self.task_contexts[task_id]
        agent_name=context["active_agent"]
        agent=self.agents[agent_name]

        input_data={
            "task_id":task_id,
            "messages":context["messages"],
            "available_tools":self._get_tool_schemas() if self._get_tool_schemas() else None
        }

        response = agent.act(input_data)
        context["step_count"] += 1

        if response.get("response") is not None:
            logger.info(f"Task {task_id} completed by {agent_name}")
            del self.task_contexts[task_id]  # 清理上下文
            return True, response["response"]

        else:
            raise ValueError("Agent response must contain 'response' or 'tool_calls'")
        

    
    def _get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            }
            for name, tool in self.tools.items()
        ]
    
    def execute_tools_and_continue(
            self,
            task_id:str,
            tool_calls:List[Dict],
            current_agent:str,
    )->Dict[str,Any]:
        results=[]
        for tc in tool_calls:
            tool_name=tc['tool']
            if tool_name not in self.tools:
                results={"error":f'Tool {tool_name} not found'}
            else:
                results=self.tool_runner.run(
                    tool=self.tools[tool_name],
                    params=tc['args'],
                    context={"task_id":task_id,'agents':current_agent}
                )
            event={
                "type":"tool_results",
                "results":results,
                "task_id":task_id
            }
            next_agent=current_agent # 简化：默认回原 Agent
            self.agents[next_agent].observe(event)

            return {"next_action":"continue","next_agent":next_agent}
        
    def shutdown(self):
        for agent in self.agents.values():
            agent.shutdown()
