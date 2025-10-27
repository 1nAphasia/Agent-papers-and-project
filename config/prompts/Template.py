import os
import json
from jinja2 import Template
from typing import Dict,Any


class JinjaPromptTemplate:
    def __init__(self,template_str):
        self.template=Template(template_str)

    def render(self,context:Dict[str,Any])->str:
        context={
            "to_json":lambda x:json.dumps(x,ensure_ascii=True,indent=2),
            **context
        }
        return self.template.render(context)