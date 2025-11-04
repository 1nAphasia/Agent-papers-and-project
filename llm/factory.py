from llm.adapter import LLMAdapter
from llm.clients.deepseek import DeepSeekLLMAdapter
from config.schema import LLMConfig

def get_llm_adapter(config:LLMConfig)->LLMAdapter:
    print(config.provider)
    if config.provider == "deepseek" :
        return DeepSeekLLMAdapter(config)