from app import AgentSystem
from agent.manager import AgentManager
from config import settings
from tools.mcp_client import MCPClient
from agent.planneragent import PlannerAgent
from agent.faissmanageagent import AsyncAgent
from config.logger import setup_global_logger, get_logger
import asyncio
from doc_process.documentprocessor import DocumentProcessor


setup_global_logger()
logger = get_logger(__name__)


async def main():
    # 创建AgentSystem
    client = MCPClient()
    await client.connect_to_server(r"mcp_server\new_server.py")
    manager = AgentManager()
    faissmanager_cfg = settings.agents.get("faissmanager", {})
    manager.register_agent(
        "faissmanager",
        AsyncAgent(
            name="faissmanager", role=faissmanager_cfg.role, config=settings.llm
        ),
    )

    manager.register_mcp_client("weatherClient", client)
    system = AgentSystem(manager)
    # 为处理文档单独调用一些client的功能。
    dp = DocumentProcessor()
    all_processed = dp.process_directory(dir_path="docs")
    result_tuple = await client.call_tools("add_documents", all_processed)
    logger.info(f"尝试向服务器添加文档,服务器返回结果:{str(result_tuple)}")
    # 测试任务
    # task=r""
    # logger.info("开始运行任务,任务为："+task)
    # response = await system.run_task(task)
    # print("✅ 结果:", response)

    # system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
