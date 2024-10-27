import PIL.Image
from ..config import IMAGE_PIL_MAX_PIXELS_INT

PIL.Image.MAX_IMAGE_PIXELS = IMAGE_PIL_MAX_PIXELS_INT  # 抑制PIL大文件警告