from typing import Dict, Any
from .prompts import *

class LLMConfig:
    def __init__(self, input_dict:Dict[str, Any]):
        assert 'model' in input_dict.keys()
        assert 'base_url' in input_dict.keys()
        

        self.model = input_dict['model']
        self.base_url = input_dict['base_url']

        self.api_key = input_dict['api_key'] if 'api_key' in input_dict.keys() else 'EMPTY'
        self.generation_config = input_dict['generation_config'] if 'generation_config' in input_dict.keys() else {}
        self.stop_condition = input_dict['stop_condition'] if 'stop_condition' in input_dict.keys() else None
        self.tool_condition = input_dict['tool_condition'] if 'tool_condition' in input_dict.keys() else None
        self.is_debug = input_dict['is_debug'] if 'is_debug' in input_dict.keys() else None


class GlobalConfig:
    def __init__(self, ):
        self.__dict__={
            "DEEPSEEK_CONFIG":{
                "base_url":r'https://api.deepseek.com',
            },
            "PLAIN_PROMPT":{
                "user_prompt": PlainPrompt_User_Template,
                "assistant_prefix": PlainPrompt_User_Template
            }
        }