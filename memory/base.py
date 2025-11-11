# memory/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseMemory(ABC):
    """记忆基类,定义标准接口"""
    
    @abstractmethod
    def add(self, key: str, value: Any) -> None:
        """存储记忆"""
        pass
        
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """检索记忆"""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """语义检索相关记忆"""
        pass

class SimpleMemory(BaseMemory):
    """简单的内存字典实现"""
    def __init__(self):
        self.store: Dict[str, Any] = {}
        
    def add(self, key: str, value: Any) -> None:
        self.store[key] = value
        
    def get(self, key: str) -> Optional[Any]:
        return self.store.get(key)
        
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        # 简单实现,后续可扩展为向量检索
        return list(self.store.items())[:k]