
from tqdm.auto import tqdm
import os.path

from ..utils import resize_and_save_image
from ..config import RESIZE_OUTPUTPATH

class ResizeImage:
    def __init__(self, save_dir = None, overwrite=False, max_size: int = 2048, min_size: int = 1024, bucket = {}):
        if save_dir is None:
            self.save_dir = RESIZE_OUTPUTPATH
        else:
            self.save_dir = save_dir
        self.overwrite = overwrite
        self.min_size = min_size
        self.max_size = max_size
        self.bucket = bucket

    def run(self, main_value):
        # self._validate_input(main_value)  # Uncomment if validation is needed
        resized_images = self._resize_images(main_value)
        return resized_images

    def _resize_images(self, image_dict):
        resized_images = {}

        # Use tqdm to display a progress bar
        for relative_path, value in tqdm(image_dict.items(), desc="Resizing Images", unit="image"):
            image_path = value['absolute_path']
            image_dir = value['origin_dir']

            resize_and_save_image(image_path=image_path, 
                        save_dir=self.save_dir, 
                        overwrite = self.overwrite,
                        image_dir = image_dir,
                        max_size = self.max_size, 
                        min_size = self.min_size, 
                        buckets = self.bucket)
            image_dict[relative_path]['absolute_path'] = os.path.join(self.save_dir,relative_path)
            image_dict[relative_path]['save_dir'] = self.save_dir
        
        return resized_images
