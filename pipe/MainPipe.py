import os.path
import glob
from ..config import *

class MainPipe:
    def __init__(self, dataset_paths):
        if type(dataset_paths) == list:
            self.dataset_paths = dataset_paths
        elif type(dataset_paths) == str:
            self.dataset_paths = [dataset_paths]
        
        self.main_value = self.get_images()
        self.pipes = []

    def get_images(self):
        image_dict = {}
        
        for dataset_path in self.dataset_paths:
            absolute_image_dir = os.path.abspath(dataset_path)
            # 使用 glob 查找所有图像文件
            for ext in IMAGE_EXTENSIONS_LIST:
                for file_path in glob.glob(os.path.join(dataset_path, '**', '*' + ext), recursive=True):
                    relative_path = os.path.relpath(file_path, dataset_path)
                    absolute_path = os.path.join(absolute_image_dir, relative_path)
                    image_dict[relative_path.replace('./', '')] = {
                        'origin_dir': absolute_image_dir,
                        'absolute_path': absolute_path, 
                    }
        return image_dict

    def add(self, pipe):
        self.pipes.append(pipe)

    def run(self, *pipes):
        # 将所有传入的管道添加到 pipes 列表中
        if pipes:
            self.pipes = []
            for pipe in pipes:
                self.add(pipe)

        self.process_pipes()

    def process_pipes(self):
        for pipe in self.pipes:
            if hasattr(pipe, 'run'):
                batch = pipe.run(self.main_value)
            else:
                raise ValueError(f"Pipe {pipe} does not have a run method")

        self.main_value.update(batch)

# 使用示例
# pipe1 = ...  # 创建管道1的实例
# pipe2 = ...  # 创建管道2的实例
# pipe3 = ...  # 创建管道3的实例

# image_dir = "path/to/images"
# main_pipe = MainPipe(image_dir)
# main_pipe.run(pipe1, pipe2, pipe3)