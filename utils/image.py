from .os import get_var, set_env
from ..config import *
from ..utils import i18n
from ..utils import logger, logger_i18n
from pathlib import Path
from typing import Union, List

__var__ = ["IMAGE_EXTENSIONS_SET"]

__fucn__ = ["find_images_in_directory",
           "logging_with_none_image",
           "filter_images_with_txt"]

__all__ = __var__ + __fucn__

IMAGE_EXTENSIONS_SET = get_var("IMAGE_EXTENSIONS_SET", IMAGE_EXTENSIONS_SET)
set_env("IMAGE_EXTENSIONS_SET", IMAGE_EXTENSIONS_SET)

def find_images_in_directory(directory: Path, 
                             recursive: bool = True):
    images = []
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS_SET:
            images.append(item)
        elif item.is_dir() and recursive:
            images.extend(find_images_in_directory(item, recursive))
    return images

def logging_with_none_image(image_paths: List[Path], 
                            raise_error: bool = False):
    if len(image_paths) == 0:
        if raise_error:
            raise ValueError(i18n("No images found"))
        else:
            logger_i18n.warning("No images found")

def filter_images_with_txt(image_paths: List[Path]):
    image_paths_with_caption = [path for path in image_paths if not Path(path).with_suffix(".txt").exists()]
    logger_i18n.info(
            "existing $$count$$ captions",
            {
                "$$count$$": len(image_paths_with_caption)
            }
    )
    return image_paths - image_paths_with_caption
