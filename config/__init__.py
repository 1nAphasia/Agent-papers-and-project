from .loader import load_config
from .schema import GlobalConfig
from typing import Optional
import os

# 单例 settings，应用启动时导入会加载默认路径（可由 env 指定 CONFIG_PATH）
current_dir = os.path.dirname(os.path.abspath(__file__))+'/defaults.py'
settings: GlobalConfig = load_config(path=current_dir)
__all__ = ["load_config", "settings", "GlobalConfig"]