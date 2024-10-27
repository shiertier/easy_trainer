import os.path
from os import makedirs
from loguru import logger
import sys
from ..config import *
from .translater import i18n
from typing import Dict, Any

__all__ = ["logger", "logger_i18n"]

makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "file_{time}.log")

logger.remove()
logger.add(log_file_path, rotation="1 MB", level=LOG_LEVEL_STR)  # 每个文件最大1MB

if LOG_SYS_BOOL:
    logger.add(sys.stdout, level=LOG_LEVEL_STR)

class Logger_I18n:
    def __init__(self, base_logger):
        self.base_logger = base_logger

    def _translate(self, message: str, replace_dict: Dict[str, Any] = None) -> str:
        """Translates a message and replaces placeholders if a dictionary is provided."""
        translated_message = i18n(message)  # Use the existing i18n function
        if replace_dict:
            for key, value in replace_dict.items():
                translated_message = translated_message.replace(key, value)
        return translated_message

    def info(self, message: str, replace_dict: Dict[str, Any] = None) -> None:
        """Logs an info message with translation support."""
        translated_message = self._translate(message, replace_dict)
        self.base_logger.info(translated_message)

    def debug(self, message: str, replace_dict: Dict[str, Any] = None) -> None:
        """Logs a debug message with translation support."""
        translated_message = self._translate(message, replace_dict)
        self.base_logger.debug(translated_message)

    def warning(self, message: str, replace_dict: Dict[str, Any] = None) -> None:
        """Logs a warning message with translation support."""
        translated_message = self._translate(message, replace_dict)
        self.base_logger.warning(translated_message)

    def error(self, message: str, replace_dict: Dict[str, Any] = None) -> None:
        """Logs an error message with translation support."""
        translated_message = self._translate(message, replace_dict)
        self.base_logger.error(translated_message)

logger_i18n = Logger_I18n(logger)