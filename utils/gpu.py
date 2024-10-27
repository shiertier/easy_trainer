import torch.cuda
from ..utils import logger_i18n, i18n
from ..config import *

__all__ = ['convert_to_device_map', "is_single_gpu", "convert_to_device_list"]

def verify_gpu_input_right(gpu_ids: str):
    if gpu_ids.strip().lower() == "all":
        pass
    else:
        valid_chars = set("0123456789, ")
        if not all(char in valid_chars for char in gpu_ids):
            logger_i18n.error("Input error: Please use a proper format like '0,1,2', 'all', or a single available GPU id.")
            raise ValueError(i18n("Invalid GPU ID format."))
    return True

def convert_to_device_map(gpu_ids: str):
    devices = ','.join(convert_to_device_list(gpu_ids))
    return devices

def is_single_gpu(devices: str):
    return devices.count(',') == 0

def convert_to_device_list(gpu_ids: str):
    if gpu_ids.strip().lower() == "all":
        # 返回所有可用 GPU
        available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        verify_gpu_input_right(gpu_ids)
        available_gpus = []
        for gpu_id in gpu_ids.split(','):
            gpu_id = gpu_id.strip()  # 清理多余的空格
            if torch.cuda.is_available() and int(gpu_id) < torch.cuda.device_count():
                available_gpus.append(f"cuda:{gpu_id}")
            else:
                logger_i18n.error(
                    "GPU $$gpu_id$$ is not available or is out of range.",
                    {
                        "$$gpu_id$$": gpu_id
                    }
                )
    return available_gpus