import os
from config.prompts.Template import JinjaPromptTemplate

def load_prompt_template(agent_type:str)->JinjaPromptTemplate:
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    template_path=cur_dir+f'/templates/{agent_type}.jinja'
    if not os.path.exists(template_path):
        raise FileNotFoundError(f'Prompt template not found:{template_path}')
    with open(template_path,encoding='utf-8') as f:
        return JinjaPromptTemplate(f.read())