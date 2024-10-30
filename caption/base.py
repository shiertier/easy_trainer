from pathlib import Path
from ..utils import *

SCRIPT_DIR = Path(__file__).parent

class CaptionBase:
    def __init__(self,
                 model_name,
                 devices_key = "all",
                 local = True,
                 download = True,
                 ):
        self.model_name = model_name
        self.devices = convert_to_device_list(devices_key)
        self.is_load_model = False
        self.model_urls = {}
        self.is_download_model = download

    def is_model_exists_in_path(self):
        # 查看是否存在路径或模型文件
        if os.path.exists(self.model_name):
            return True
        else:
            logger_i18n.info("Model path does not exist / Input name is not a local path")
            return False

    def download_model(self):
        # TODO
        pass

    def load_model(self):
        pass

    def load_model_in_one_gpu(self, device):
        pass

    def generlate_base(self):
        pass

    def generate_captions(self):
        pass

    def generate_caption(self):
        pass

    def create_service(self):
        pass

    def generate_caption_remote(self):
        pass
