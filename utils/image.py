from os import PathLike,makedirs
import os.path
from pathlib import Path
from typing import Union, List
from typing import BinaryIO, List, Optional
from PIL import Image
from os import PathLike
from typing import Union, BinaryIO, List, Tuple

from .os import get_var, set_env
from ..config import *
from ..utils import i18n
from ..utils import logger, logger_i18n


__var__ =  ["IMAGE_EXTENSIONS_LIST",
            "ImageTyping",
            "MultiImagesTyping"
            ]

__fucn__ = ["find_images_in_directory",
            "logging_with_none_image",
            "filter_images_with_txt",
            "load_image",
            "load_images",
            "resize_image",
            "resize_and_save_image",
            "add_background_for_rgba",
            ]

__all__ = __var__ + __fucn__

IMAGE_EXTENSIONS_LIST = get_var("IMAGE_EXTENSIONS_LIST", IMAGE_EXTENSIONS_LIST)
set_env("IMAGE_EXTENSIONS_LIST", IMAGE_EXTENSIONS_LIST)

ImageTyping = Union[str, PathLike, bytes, bytearray, BinaryIO, Image.Image]
MultiImagesTyping = Union[ImageTyping, List[ImageTyping], Tuple[ImageTyping, ...]]

def find_images_in_directory(directory: Path, 
                             recursive: bool = True):
    images = []
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS_LIST:
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
    image_paths_with_caption = [path for path in image_paths if Path(path).with_suffix(".txt").exists()]
    logger_i18n.info(
            "existing $$count$$ captions",
            {
                "$$count$$": len(image_paths_with_caption)
            }
    )
    image_paths_set = set(image_paths)
    image_paths_with_caption_set = set(image_paths_with_caption)
    difference = image_paths_set - image_paths_with_caption_set
    return list(difference)

def _is_readable(obj):
    return hasattr(obj, 'read') and hasattr(obj, 'seek')

def _has_alpha_channel(image: Image.Image) -> bool:
    return any(band in {'A', 'a', 'P'} for band in image.getbands())

def load_image(image: ImageTyping, mode=None, force_background: Optional[str] = 'white'):
    
    if isinstance(image, (str, PathLike, bytes, bytearray, BinaryIO)) or _is_readable(image):
        image = Image.open(image)
    elif isinstance(image, Image.Image):
        pass  # just do nothing
    else:
        raise TypeError(f'Unknown image type - {image!r}.')

    if _has_alpha_channel(image) and force_background is not None:
        image = add_background_for_rgba(image, force_background)

    if mode is not None and image.mode != mode:
        image = image.convert(mode)

    return image

def load_images(images: MultiImagesTyping, mode=None, force_background: Optional[str] = 'white') -> List[Image.Image]:
    if not isinstance(images, (list, tuple)):
        images = [images]

    return [load_image(item, mode, force_background) for item in images]

def add_background_for_rgba(image: ImageTyping, background: str = 'white'):
    from .layer import istack
    return istack(background, image).convert('RGB')

def resize_image(image, 
                 max_size: int = 2048, 
                 min_size: int = 1024, 
                 buckets: dict = None):
    
    if not buckets:
        buckets = RESIZE_BUCKETS
    try:
        image = load_image(image)
    except OSError as e:
        logger_i18n("Error processing image $$image_path$$: $$e$$",{"$$image_path$$": image_path, "$$e$$":e})
    
    width, height = image.size
    total_resolution = width * height
    
    def get_size_key():
        for key in buckets.keys():
            if int(key) > min_size:
                if total_resolution >= int(key) * 0.95:
                    if int(key) > max_size:
                        return max_size
                    else:
                        return key
        return None
    
    size_key = get_size_key()
    
    if size_key is None:
        return None
    
    def get_bucket_size():
        closest_pair = min(buckets[size_key], key=lambda pair: abs(width / height - pair[0] / pair[1]))
        ratio = max(closest_pair[0] / width, closest_pair[1] / height)
        new_width = int(ratio * width)
        new_height = int(ratio * height)
        crop_x = (new_width - closest_pair[0]) // 2
        crop_y = (new_height - closest_pair[1]) // 2
        return new_width, new_height, crop_x, crop_y
    
    new_width, new_height, crop_x, crop_y = get_bucket_size()
    new_width = (new_width // 2) * 2  # 避免单数
    new_height = (new_height // 2) * 2  # 避免单数
    
    with image.resize((new_width, new_height), resample=Image.BICUBIC) as img_new:
        cropped_image = img_new.crop((crop_x, crop_y, new_width - crop_x, new_height - crop_y))
    
    return cropped_image

def resize_and_save_image(image_path: str,
                    save_dir: str,
                    overwrite: bool = False,
                    image_dir: str = None,
                    max_size: int = 2048, 
                    min_size: int = 1024, 
                    buckets: dict = None):

    # 获取文件名
    if image_dir:
        relative_path = os.path.relpath(image_path, image_dir)
        base_name = os.path.splitext(relative_path)[0]
    else:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    lossless = os.path.getsize(image_path) > 1024 ** 2
    save_path = os.path.join(save_dir, f"{base_name}.{'webp' if lossless else 'jpg'}")
    
    if not overwrite and os.path.exists(save_path):
        return 

    makedirs(os.path.dirname(save_path), exist_ok=True)

    crop_image = resize_image(image_path, max_size, min_size, buckets)

    if lossless:
        crop_image.save(save_path, format='WEBP', quality=100, lossless=True)
    else:
        crop_image.save(save_path, format='JPEG', quality=92)  # 保存为JPEG 92

    crop_image.close()
