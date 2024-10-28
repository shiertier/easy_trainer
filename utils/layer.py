from typing import Union, Tuple, Optional
import numpy as np
from PIL import ImageColor, Image
from .image import load_image, ImageTyping

__all__ = [
    'istack',
]

def _load_image_or_color(image) -> Union[str, Image.Image]:
    if isinstance(image, str):
        try:
            _ = ImageColor.getrgb(image)
        except ValueError:
            pass
        else:
            return image

    return load_image(image, mode='RGBA', force_background=None)

def _process(item):
    if isinstance(item, tuple):
        image, alpha = item
    else:
        image, alpha = item, 1

    return _load_image_or_color(image), alpha

_AlphaTyping = Union[float, np.ndarray]

def _add_alpha(image: Image.Image, alpha: _AlphaTyping) -> Image.Image:
    data = np.array(image.convert('RGBA')).astype(np.float32)
    data[:, :, 3] = (data[:, :, 3] * alpha).clip(0, 255)
    return Image.fromarray(data.astype(np.uint8), mode='RGBA')

def istack(*items: Union[ImageTyping, str, Tuple[ImageTyping, _AlphaTyping], Tuple[str, _AlphaTyping]],
           size: Optional[Tuple[int, int]] = None) -> Image.Image:
    if size is None:
        height, width = None, None
        items = list(map(_process, items))
        for item, alpha in items:
            if isinstance(item, Image.Image):
                height, width = item.height, item.width
                break
    else:
        width, height = size

    if height is None:
        raise ValueError('Unable to determine image size, please make sure '
                         'you have provided at least one image object (image path or PIL object).')

    retval = Image.fromarray(np.zeros((height, width, 4), dtype=np.uint8), mode='RGBA')
    for item, alpha in items:
        if isinstance(item, str):
            current = Image.new("RGBA", (width, height), item)
        elif isinstance(item, Image.Image):
            current = item
        else:
            assert False, f'Invalid type - {item!r}. If you encounter this situation, ' \
                          f'it means there is a bug in the code. Please contact the developer.'  # pragma: no cover

        current = _add_alpha(current, alpha)
        retval.paste(current, mask=current)

    return retval