from typing import Any,Dict,Optional
import os
import json
import yaml
from .defaults import DEFAULT_CONFIG
from .schema import GlobalConfig


def _load_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith(('.yml', '.yaml')):
            return yaml.safe_load(f) or {}
        elif path.endswith('.json'):
            return json.load(f)
        else:
            # 尝试 JSON 解析或返回空
            try:
                return json.load(f)
            except Exception:
                return {}

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: Optional[str] = None) -> GlobalConfig:
    """
    加载path中的config,如果没有,就返回默认值
    返回已验证的 GlobalConfig 实例
    """
    cfg = dict(DEFAULT_CONFIG)
    if path:
        file_cfg = _load_file(path)
        cfg = merge_dicts(cfg, file_cfg)
    # 使用 pydantic 验证
    return GlobalConfig.model_validate(cfg)