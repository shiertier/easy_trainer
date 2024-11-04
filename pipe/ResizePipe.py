#from tqdm.auto import tqdm
import os.path
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..utils import resize_and_save_image
from ..config import RESIZE_OUTPUTPATH

class ResizeImage:
    def __init__(self, save_dir=None, overwrite=False, max_size: int = 2048, min_size: int = 1024, bucket={}):
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

        # Use ProcessPoolExecutor to parallelize the resizing process
        with ProcessPoolExecutor() as executor:
            futures = []
            for relative_path, value in image_dict.items():
                image_path = value['absolute_path']
                image_dir = value['origin_dir']

                # Submit the resizing task to the executor
                future = executor.submit(resize_and_save_image, 
                                         image_path=image_path, 
                                         save_dir=self.save_dir, 
                                         overwrite=self.overwrite,
                                         image_dir=image_dir,
                                         max_size=self.max_size, 
                                         min_size=self.min_size, 
                                         buckets=self.bucket)
                futures.append((relative_path, future))

            # Use tqdm to display a progress bar
            # for relative_path, future in tqdm(as_completed(futures), total=len(futures), desc="Resizing Images", unit="image"):
            for relative_path, future in as_completed(futures):
                try:
                    future.result()  # Wait for the result of the future
                    value = image_dict[relative_path]
                    value['absolute_path'] = os.path.join(self.save_dir, relative_path)
                    value['save_dir'] = self.save_dir
                    resized_images[relative_path] = value
                except Exception as e:
                    print(f"Error processing image {relative_path}: {e}")

        return resized_images
