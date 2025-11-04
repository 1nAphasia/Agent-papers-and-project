import logging
import sys
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from config import settings

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 支持额外字段
        if hasattr(record, "task_id"):
            log_obj["task_id"] = record.task_id
        if hasattr(record, "agent"):
            log_obj["agent"] = record.agent
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)

def setup_global_logger(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    json_format: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
):
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # 清理旧 handler
    root_logger.handlers.clear()

    # 选择格式
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # 终端输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件输出
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

def get_logger(name: str = None):
    """获取全局logger或子logger"""
    return logging.getLogger(name)

# 初始化全局日志（只需调用一次）
setup_global_logger(
    log_level=settings.logging.level,
    log_file="logs/app.log",
    json_format=settings.logging.json_format
)