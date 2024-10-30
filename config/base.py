# 获取当前脚本所在目录
from pathlib import Path

CONFIG_DIR = Path(__file__).parent
TRAINER_DIR = CONFIG_DIR.parent
MODELS_DIR = TRAINER_DIR / "models"
CAPTION_MODEL_DIR = MODELS_DIR / "caption"
AESTHETIC_MODEL_DIR = MODELS_DIR / "aesthetic"
SD15_MODEL_DIR = MODELS_DIR / "sd15"
SDXL_MODEL_DIR = MODELS_DIR / "sdxl"
SD3_MODEL_DIR = MODELS_DIR / "sd3"
FLUX_MODEL_DIR = MODELS_DIR / "flux"


