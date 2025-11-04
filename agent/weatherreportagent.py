import json
from agent.base import BaseAgent
from llm.factory import get_llm_adapter
from config.prompts.loader import load_prompt_template
from config.logger import get_logger

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

        context={
            "role":self.role,
            'user_task':input_data.get('task',''),
            }
        prompts=self.prompt_template.render(context)
        messages=[{'role':'system','content':prompts}]+input_data['messages']
        logger.info("本次输入："+str(messages)+"。提供的工具包括："+str(input_data['available_tools']))

        response=self.llm.chat(
            messages=messages,
            tools=input_data['available_tools'],
            tool_choice='required',
            stream=False,
            )

        if response.finish_reason == "tool_calls":
            logger.info("agent进行一次工具调用")
            return {
                "tool_calls":[
                {
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments)
                } for tc in response.message.tool_calls or []
            ]
            }
        elif response.finish_reason == "stop":
            logger.info("agent返回了文本")
            return {
                "content":response.message.content
            }

        
    def shutdown(self):
        return 