from typing import Dict,Any
from abc import ABC,abstractmethod


class BaseTool(ABC):
    def __init__(self,name):
        self.name=name
    @abstractmethod
    def call(self,params:Dict[str,Any])->Dict[str,Any]:
        return NotImplementedError("工具需实现call方法")