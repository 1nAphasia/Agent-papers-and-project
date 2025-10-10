import requests
from typing import Dict, List, Any
from openai import OpenAI
import re
import json
from termcolor import colored
import os
import sys
import time
import multiprocessing
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
# from hipporag import HippoRAG
from copy import deepcopy



current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

# from configs import CommonConfig
from config.configs import LLMConfig
# from llm_agent.context import BaseContextManager
# from llm_agent.tools.tool_manager import BaseToolManager

class BaseAgent:
    def __init__(self,llm_config:Dict[str,Any]):
        
        self.llm_config:LLMConfig=LLMConfig(llm_config)
        self.client=OpenAI(base_url=self.llm_config.base_url,api_key=self.llm_config.api_key)

        # assert self.rag_chunk>self.rag_overlapping

    def extract_tool_content(self,content):
        if not self.llm_config.tool_condition:
            return content, ''
        matches = list(re.finditer(self.llm_config.tool_condition, content, re.DOTALL))
        detected_num = len(matches)
        
        if detected_num > 0:
            match = matches[0]
            code_content = match.group(1)
            match_start_index = match.start()
            cut_text = content[:match_start_index]

            return cut_text, code_content

        return content, ''

    def call_api(self,prompt:str,enable_rag=True):
        try:
            messages=[{"role":'user',"content":prompt}]
            with self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=messages,
                stream=True,
                **self.llm_config.generation_config
            ) as stream:
                full_response=""

                for chunk in stream:
                    content=chunk.choices[0].delta.content
                    if content is not None:
                        if self.llm_config.is_debug:
                            print(content,end='',flush=True)
                        full_response+=content

        except KeyboardInterrupt:
            print('KeyboardInterrupt')

        except Exception as e:
            import traceback
            print(f"error {traceback.format_exc()},Model:{self.llm_config.model}")

        return {
            'content':full_response.strip(),
            'type':'full_text',
        }
    
    def step(self,input_prompt:str):
        api_call_count=1
        step_response_content=''
        while api_call_count<=(1):
            step_response_dict=self.call_api(input_prompt,enable_rag=api_call_count)
            api_call_count+=1
            step_response_content+=step_response_dict['content']

            step_response_type=step_response_dict['type']
            if step_response_type in ['full_text']:
                break
            else:
                input_prompt+= step_response_content
                if self.llm_config.is_debug:
                    print(colored(f'\n\n[Continue generation]\n{input_prompt}','cyan',attrs=['bold']))
            
        agent_response,tool_call_content=self.extract_tool_content(step_response_content)

        return{
            'step_response': agent_response,
            'tool_call_content': tool_call_content
        }
    

if __name__ =="__main__":
    from multiprocessing import freeze_support
    freeze_support()

    cfg={
        'model':'deepseek-reasoner',
        'base_url':r'https://api.deepseek.com',
        'api_key':'sk-716f93e2299940b58b5838939cade9d0',
        'generation_config': {
            'max_tokens':5000,
            'temperature':0.5
            },
        'stop_condition': r'<code[^>]*>((?:(?!<code).)*?)</code>',
        'tool_condition': r'<code[^>]*>((?:(?!<code).)*?)</code>',
        'is_debug':True
        }
    llm_config=LLMConfig(cfg)
    base_agent=BaseAgent(cfg)
    base_agent.step("浓硫酸的化学性质有哪些？")