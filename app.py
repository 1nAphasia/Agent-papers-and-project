# app.py
from agent.manager import AgentManager
from agent.planneragent import PlannerAgent
from config import settings

class AgentSystem:
    
    def __init__(self, manager: AgentManager):
        self.manager = manager

    def run_task(self, user_input: str) -> str:
        """执行单个用户任务（简化版）"""
        task_id = self.manager.start_task("planner", user_input)
        while True:
            is_done, response = self.manager.step(task_id)
            if is_done:
                return response

    def shutdown(self):
        self.manager.shutdown()

def create_agent_system() -> AgentSystem:
    """工厂函数：创建并配置完整的 Agent 系统"""
    manager = AgentManager()

    planner_cfg = settings.agents.get("planner", {})

    manager.register_agent(
        "planner",
        PlannerAgent(name="planner", role=planner_cfg.role, config=settings.llm)
    )

    return AgentSystem(manager)