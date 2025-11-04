# app.py
from agent.manager import AgentManager
from agent.planneragent import PlannerAgent
from agent.weatherreportagent import WeatherAgent
from config.logger import get_logger
from config import settings
from tools.mcp_client import MCPClient

logger=get_logger(__name__)


class AgentSystem:
    
    def __init__(self, manager: AgentManager):
        self.manager = manager

    def run_task(self, user_input: str) -> str:
        """执行单个用户任务（简化版）"""
        task_id = self.manager.start_task("weather", user_input)
        while True:
            is_done, response = self.manager.step(task_id)
            if is_done:
                return response

    def shutdown(self):
        self.manager.shutdown()

async def create_agent_system() -> AgentSystem:
    """工厂函数：创建并配置完整的 Agent 系统"""
    
    manager = AgentManager()

    planner_cfg = settings.agents.get("planner", {})

    manager.register_agent(
        "planner",
        PlannerAgent(name="planner", role=planner_cfg.role, config=settings.llm)
    )

    weather_cfg = settings.agents.get("weather", {})
    client=MCPClient()
    await client.connect_to_server(r"C:\Users\Administrator\Desktop\Research\Agent-papers-and-project\mcp_server\server.py")
    logger.info("server should be connected.")
    manager.register_agent("weather",WeatherAgent(name="weather",role=weather_cfg.role,config=settings.llm))
    manager.register_mcp_client("weatherClient",client)

    return AgentSystem(manager)