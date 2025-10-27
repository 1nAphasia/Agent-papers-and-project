import json
from agent.base import BaseAgent
from llm.factory import get_llm_adapter
from config.prompts.loader import load_prompt_template


class PlannerAgent(BaseAgent):
    def __init__(self, name, role = '', config = None):
        super().__init__(name, role, config)

    def initialize(self):
        return super().initialize()
    
    def _setup(self):
        self.llm=get_llm_adapter(self.config)
        self.prompt_template=load_prompt_template(self.role)
    
    def act(self, input_data):
        context={
            "role":self.role,
            'user_task':input_data.get('task',''),
            }
        prompts=self.prompt_template.render(context)
        messages=[{'role':'system','content':prompts}]+input_data['messages']
        response=self.llm.chat(
            messages=messages,
            tools=None,
            tool_choice='none',
            stream=False,
            )
        return {
            "response": response["content"],
            "tool_calls": None,
            "metadata": {}
        }
    def shutdown(self):
        return 