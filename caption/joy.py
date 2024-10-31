from pathlib import Path
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TVF

from .._class.dataset import ImageDataset
from ..utils import *
from ..config import *
from .base import CaptionBase
import os.path
import copy

SCRIPT_DIR = Path(__file__).parent

class Joy2(CaptionBase):
    def __init__(self,
                 model_name='llama-joycaption-alpha-two-hf-llava',
                 devices_key = "all",
                 local = True,
                 download = True,
                 ):
        self.model_name = model_name
        self.devices = convert_to_device_list(devices_key)
        self.is_load_tokenizer = False
        self.is_load_llava_model_cpu = False
        self.models = {}
        self.model_urls = {
                            'llama-joycaption-alpha-two-hf-llava': 
                            {
                                'type': 'repo',
                                'url': 'https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava'
                            }
                        }
        
        if download:
            self.download_model()

        if local:
            self.load_model()
    
    def _verify_model_name(self):
        if self.model_name not in self.model_urls:
            raise ValueError(i18n("Model name $$self.model_name$$ not found in model_urls",{"$$self.model_name$$": self.model_name}))

    def download_model(self, 
                       local_dir = None):
        self._verify_model_name(self)
        download_data = self.model_urls[self.model_name]
        download_type = download_data['type']
        download_url = download_data['url']
        name = download_url.split('/')[-1]
        save_dir = os.path.join(CAPTION_MODEL_DIR, name)
        if not os.path.exists(save_dir):
            if local_dir is None:
                download_huggingface_model(download_url, repo_type=download_type, local_dir=CAPTION_MODEL_DIR)
            else:
                download_huggingface_model(download_url, repo_type=download_type, local_dir=local_dir)
        else:
            logger_i18n.info("model already exists, if model load failed, please delete the model folder and try again")

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        is_valid_tokenizer = isinstance(self.tokenizer, PreTrainedTokenizer) or isinstance(self.tokenizer, PreTrainedTokenizerFast)
        tokenizer_valid_error_info = i18n("Tokenizer is of type $$tokenizer_type$$", {"$$tokenizer_type$$": type(self.tokenizer)})
        assert is_valid_tokenizer, tokenizer_valid_error_info
        self.end_of_header_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        self.end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        assert isinstance(self.end_of_header_id, int) and isinstance(self.end_of_turn_id, int)
        logger_i18n.info("joycaption2: load tokenizer done")
        self.is_load_tokenizer = True
    
    def load_model_in_cpu(self):
        if not self.is_load_tokenizer:
            self.load_tokenizer()
        if not self.is_load_llava_model_cpu:
            self.llava_model_cpu = LlavaForConditionalGeneration.from_pretrained(JOY2_MODEL_STR, 
                                                                                 torch_dtype="bfloat16")
            self.image_token_id = self.llava_model_cpu.config.image_token_index,
            self.image_seq_length = self.llava_model_cpu.config.image_seq_length
            self.is_load_llava_model_cpu = True

    def load_model_in_gpu(self,gpu_id=0):
        if gpu_id in list(self.model.keys()):
            logger_i18n.debug("joycaption2: model is already loaded")
            return
        
        self.llava_model_cpu = self.load_model_in_cpu()

        model_copy = copy.deepcopy(self.llava_model_cpu)
        model_copy.to(torch.device(f'cuda:{gpu_id}'))

        self.model[gpu_id] = {"status": 0}
        self.model[gpu_id] = {
            "model": model_copy,
            "vision_dtype": model_copy.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype,
            "vision_device": model_copy.vision_tower.vision_model.embeddings.patch_embedding.weight.device,
            "language_device": model_copy.language_model.get_input_embeddings().weight.device,
            "status": 1
        }
        
        logger_i18n.info("joycaption2: load model done in gpu: $$gpu_id$$", {"$$gpu_id$$": gpu_id})

    def load_model(self):
        for gpu_id in self.devices:
            self.load_model_in_gpu(gpu_id)
        del self.llava_model_cpu

    def find_images_by_dir(self, image_dir, filter_captoion=True, recursive=True):
        image_paths = find_images_in_directory(image_dir, recursive = recursive)
        if filter_captoion:
            return filter_images_with_txt(image_paths)
        else:
            return image_paths

    def images_dataloader(self, prompt, image_paths, batch_size, num_workers=1):
        dataset = ImageDataset(prompt = prompt,             
                               paths = image_paths,             
                               tokenizer = self.tokenizer,             
                               image_token_id = self.image_token_id,             
                               image_seq_length = self.image_seq_length)
        dataloader = DataLoader(dataset, 
                                collate_fn=dataset.collate_fn, 
                                num_workers=num_workers, 
                                shuffle=False, 
                                drop_last=False, 
                                batch_size=batch_size)
        return dataloader

    def generate_base(self, gpu_key, data, max_tokens, temperature, top_k, top_p, is_greedy):
        pixel_values = data['pixel_values'].to(self.model[gpu_key]['vision_device'], non_blocking=True)
        input_ids = data['input_ids'].to(self.model[gpu_key]['language_device'], non_blocking=True)
        attention_mask = data['attention_mask'].to(self.model[gpu_key]['language_device'], non_blocking=True)

        # Normalize the image
        pixel_values = pixel_values / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(self.model[gpu_key]['vision_dtype'])

        # debug
        logger_i18n.debug("input_ids shape:")
        logger.debug(input_ids.shape)
        logger_i18n.debug("pixel_values shape:")
        logger.debug(pixel_values.shape)
        logger_i18n.debug("attention_mask shape:")
        logger.debug(attention_mask.shape)

        # Generate the captions
        generate_ids = self.llava_model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=not is_greedy,
            suppress_tokens=None,
            use_cache=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Trim off the prompts
        assert isinstance(generate_ids, torch.Tensor)
        generate_ids = generate_ids.tolist()
        generate_ids = [trim_off_prompt(ids, self.end_of_header_id, self.end_of_turn_id) for ids in generate_ids]

        # Decode the captions
        captions = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        captions = [c.strip() for c in captions]

        pixel_values.to('cpu')
        input_ids.to('cpu')
        attention_mask.to('cpu')
        del pixel_values, input_ids, attention_mask
        return captions
    
    def generate_caption_by_path(self,
                                image: str, 
                                prompt: str,
                                batch_size = 1,
                                prepend_string = '',
                                append_string = '',
                                max_tokens = 300,
                                temperature = 0.5,
                                top_p = 0.9,
                                top_k = 10,
                                is_greedy = False):
        dataloader = self.images_dataloader(prompt, [image], batch_size)
        for batch in dataloader:
            first_batch = batch
            break
        
        first_gpu_key = list(self.model.keys())[0]
        captions = self.generate_base(first_gpu_key, first_batch, max_tokens, temperature, top_k, top_p, is_greedy)
        caption_str = prepend_string + captions[0] + append_string
        logger_i18n.debug("Caption: $$caption$$", {"$$caption$$": caption_str})
        return caption

    def generate_captions_by_paths(self,
                                   image_paths,
                                   prompt,
                                   gpu_key, 
                                   batch_size = 1,
                                   prepend_string = '',
                                   append_string = '',
                                   max_tokens = 300,
                                   temperature = 0.5,
                                   top_p = 0.9,
                                   top_k = 10,
                                   is_greedy = False,
                                   overwrite = False,
                                   ):
        if not overwrite:
            image_paths = filter_images_with_txt(image_paths)
        dataloader = self.images_dataloader(prompt, image_paths, batch_size)

        for batch in dataloader:
            captions = self.generate_base(gpu_key,batch, max_tokens, temperature, top_k, top_p, is_greedy)

            for path, caption in zip(batch['paths'], captions):
                caption_str = prepend_string + caption + append_string
                write_caption(Path(path), caption_str)

    def generate_captions_by_dir(self,
                                   image_paths,
                                   prompt,
                                   batch_size = 1,
                                   prepend_string = '',
                                   append_string = '',
                                   max_tokens = 300,
                                   temperature = 0.5,
                                   top_p = 0.9,
                                   top_k = 10,
                                   is_greedy = False,
                                   overwrite = False,
                                   recursive = True,
                                   ):

        if not self.is_load_model:
            self.load_model()

        image_paths = find_images_in_directory(Path(image_dir), 
                                               recursive = recursive)


    def create_service(self, server_name, server_port):
        # TODO
        pass

      

model_name = JOY2_MODEL_STR
devices_key = JOY2_GPU_COUNT_STR
local = True
image_dir = DATASET_PIC_ORIGIN_DIR
is_image_load_recursive = IMAGE_LOAD_RECURSIVE_BOOL
overwrite_caption = CAPTION_OVERWRITE_BOOL
batch_size = JOY2_MAX_BATCH_SIZE
max_tokens = JOY2_MAX_NEW_TOKENS_INT
prepend_string = JOY2_ADD_PREPEND_STR
append_string = JOY2_ADD_APPEND_STR
is_greedy = JOY2_IS_GREEDY_BOOL