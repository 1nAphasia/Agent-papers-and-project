import json
from agent.base import BaseAgent
from llm.factory import get_llm_adapter
from config.prompts.loader import load_prompt_template
from config.logger import get_logger
from agent.response import LLMResponse

logger=get_logger(__name__)

class WeatherAgent(BaseAgent):
    def __init__(self, name, role = '', config = None):
        super().__init__(name, role, config)

    def initialize(self):
        return super().initialize()
    
    def _setup(self):
        self.llm=get_llm_adapter(self.config)
        self.prompt_template=load_prompt_template(self.role)
    
    def act(self, input_data):
        logger.info(self.role+"开始运行。任务为："+input_data.get('task',''))

        task_id = input_data.get("task_id")
        history = self.recall(f"history_{task_id}") or []

        context={
            "role":self.role,
            'user_task':input_data.get('task',''),
            "history":history
            }
        
        tool_results=self.recall(f"tool_results_{task_id}")
        if tool_results:
            messages = [
                {"role": "system", "content": self.prompt_template.render(context)},
                *history,
                {"role": "assistant", "content": "我已获取到天气数据,让我为您解析:"},
                {"role": "tool", "content": json.dumps(tool_results, ensure_ascii=False)}
            ]
            self.remember(f"tool_results_{task_id}", None)
        else:
            messages = [
                {"role": "system", "content": self.prompt_template.render(context)},
                *history
            ] + input_data["messages"]

        logger.info("本次输入："+str(messages)+"。提供的工具包括："+str(input_data['available_tools']))

        response=self.llm.chat(
            messages=messages,
            tools=input_data['available_tools'],
            tool_choice='auto',
            stream=False,
            )
        
        history.extend([
            {"role": "user", "content": input_data["messages"][-1]["content"]},
            {"role": "assistant", "content": response.choices[0].message.content}
        ])
        self.remember(f"history_{task_id}", history)

        resp=LLMResponse.from_openai(response.model_dump())

        if resp.finish_reason == "tool_calls":
            logger.info("agent进行一次工具调用")
            return resp
            
        elif resp.finish_reason == "stop":
            logger.info("agent返回了文本")
            return resp
        
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