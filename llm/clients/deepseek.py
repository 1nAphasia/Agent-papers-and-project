from llm.adapter import LLMAdapter
from openai import OpenAI
import time
from tenacity import retry,stop_after_attempt,wait_exponential
from config.logger import setup_global_logger,get_logger

setup_global_logger()
logger=get_logger(__name__)

class DeepSeekLLMAdapter(LLMAdapter):
    def __init__(self, config):
        super().__init__(config)
        self._setup_client()

    def _setup_client(self):
        self.client=OpenAI(base_url=self.config.base_url,
                           api_key=self.config.api_key,
                           timeout=self.config.timeout,
                            max_retries=self.config.max_retries
                           )
        
    @retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1,min=2,max=10))
    def chat(self, messages, tools = None, tool_choice = "auto", stream = False, **kwargs):
        start=time.time()
        try:
            response=self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                timeout=self.timeout,
                **kwargs
            )
            
            if not stream:
                choice=response.choices[0]
                duration=time.time()-start
                
                logger.info("本次调用Deepseek api耗时"+str(duration)+"秒")
                
                return choice
            else:
                return response
        except Exception as e:
            raise

    def shutdown(self):
        return 


