from app import create_agent_system
from config.logger import setup_global_logger,get_logger
import asyncio

setup_global_logger()
logger=get_logger(__name__)

async def main():
    system = await create_agent_system()
    
    # 测试任务
    task=r"纽约的天气如何？纽约的经纬度是‌北纬40.43度，西经74度。"
    logger.info("开始运行任务,任务为："+task)
    response = system.run_task(task)
    print("✅ 结果:", response)
    
    system.shutdown()

if __name__ == "__main__":  
    asyncio.run(main())
