# app.py
from agent.manager import AgentManager
from config.logger import get_logger

logger=get_logger(__name__)


class AgentSystem:
    
    def __init__(self, manager: AgentManager):
        self.manager = manager
        

    async def run_task(self, user_input: str) -> str:
        """执行单个用户任务（简化版）"""
        task_id = self.manager.start_task("weather", user_input)
        while True:
            is_done, response = await self.manager.step(task_id)
            if is_done:
                return response

    def shutdown(self):
        self.manager.shutdown()

