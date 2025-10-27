from app import create_agent_system

if __name__ == "__main__":
    system = create_agent_system()
    
    # 测试任务
    response = system.run_task("如何把大象放进冰箱？")
    print("✅ 结果:", response)
    
    system.shutdown()