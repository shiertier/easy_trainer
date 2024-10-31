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
        self.gpu_load_model_sucess = []
        self.model_urls = {
                            'llama-joycaption-alpha-two-hf-llava': 
                            {
                                'type': 'repo',
                                'url': 'https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava'
                            }
                        }
        self.models = {}
        if download:
            self.download_model()
        if local:
            self.load_model()

    def download_model(self):
        download_data = self.model_urls[self.model_name]
        download_type = download_data['type']
        download_url = download_data['url']
        name = download_url.split('/')[-1]
        save_dir = os.path.join(CAPTION_MODEL_DIR, name)
        if not os.path.exists(save_dir):
            download_huggingface_model(download_url, repo_type=download_type, local_dir=CAPTION_MODEL_DIR)
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
    
    def model_in_cpu(self):
        if not self.is_load_tokenizer:
            self.load_tokenizer()
        try:
            return self.llava_model_cpu
        except Exception:    
            self.llava_model_cpu = LlavaForConditionalGeneration.from_pretrained(JOY2_MODEL_STR, 
                                                                                 torch_dtype="bfloat16")
            return self.llava_model_cpu

    def load_model_in_gpu(self,gpu_id=0):
        if gpu_id in self.gpu_load_model_sucess:
            logger_i18n.debug("joycaption2: model is already loaded")
            return
        
        model_copy = type(self.llava_model_cpu)()
        model_copy.load_state_dict(self.llava_model_cpu.state_dict())
        model_copy.to(torch.device(f'cuda:{gpu_id}'))
        self.model[gpu_id] = {}
        self.model[gpu_id] = {
            "model": model_copy,
            "vision_dtype": model_copy.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype,
            "vision_device": model_copy.vision_tower.vision_model.embeddings.patch_embedding.weight.device,
            "language_device": model_copy.language_model.get_input_embeddings().weight.device,
            "image_token_id": model_copy.config.image_token_index,
            "image_seq_length": model_copy.config.image_seq_length
        }
        logger_i18n.info("joycaption2: load model done in gpu: $$gpu_id$$", {"$$gpu_id$$": gpu_id})

    def load_model(self):
        for gpu_id in self.devices:
            self.load_model_in_gpu(gpu_id)
        del self.llava_model_cpu

    def find_images_by_dir(self, ,recursive=True):
        image_paths = find_images_in_directory(image_dir, recursive = recursive)












    def generate_base(self, data, max_tokens, temperature, top_k, top_p, is_greedy):

        pixel_values = data['pixel_values'].to(self.vision_device, non_blocking=True)
        input_ids = data['input_ids'].to(self.language_device, non_blocking=True)
        attention_mask = data['attention_mask'].to(self.language_device, non_blocking=True)

        # Normalize the image
        pixel_values = pixel_values / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(self.vision_dtype)

        # debug
        logger_i18n.debug("input_ids shape:")
        logger.debug(input_ids.shape)
        logger_i18n.debug("pixel_values shape:")
        logger.debug(pixel_values.shape)
        logger_i18n.debug("attention_mask shape:")
        logger.debug(attention_mask.shape)

        # Generate the captions

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

        return captions

    def generate_captions_by_paths(self,
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
                                   ):
        dataset = ImageDataset(prompt = prompt,             
                               paths = image_paths,             
                               tokenizer = self.tokenizer,             
                               image_token_id = self.image_token_id,             
                               image_seq_length = self.image_seq_length)
        dataloader = DataLoader(dataset, 
                                collate_fn=dataset.collate_fn, 
                                num_workers=16, 
                                shuffle=False, 
                                drop_last=False, 
                                batch_size=batch_size)

        for batch in dataloader:
            captions = self.generate_base(batch, max_tokens, temperature, top_k, top_p, is_greedy)

            for path, caption in zip(batch['paths'], captions):
                caption_str = prepend_string + caption + append_string
                write_caption(Path(path), caption_str)


    def generate_captions_by_dir(self, 
                                image_dir, 
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
                                recursive = False):

        if not self.is_load_model:
            self.load_model()

        image_paths = find_images_in_directory(Path(image_dir), 
                                               recursive = recursive)
        logger_i18n.info("Found $$count$$ images.",{"$$count$$": len(image_paths)})
        logging_with_none_image(image_paths, 
                                raise_error = False)
        if not overwrite:
            image_paths = filter_images_with_txt(image_paths)

        dataset = ImageDataset(prompt = prompt,             
                               paths = image_paths,             
                               tokenizer = self.tokenizer,             
                               image_token_id = self.image_token_id,             
                               image_seq_length = self.image_seq_length)
        dataloader = DataLoader(dataset, 
                                collate_fn=dataset.collate_fn, 
                                num_workers=16, 
                                shuffle=False, 
                                drop_last=False, 
                                batch_size=batch_size)

        pbar = tqdm(total=len(image_paths), desc=i18n("Captioning images..."), dynamic_ncols=True)

        for batch in dataloader:
            captions = self.generate_base(batch, max_tokens, temperature, top_k, top_p, is_greedy)

            for path, caption in zip(batch['paths'], captions):
                caption_str = prepend_string + caption + append_string
                write_caption(Path(path), caption_str)
            
            pbar.update(len(captions))

    def generate_caption_by_path(   self, 
                            image, 
                            prompt,
                            batch_size = 1,
                            prepend_string = '',
                            append_string = '',
                            max_tokens = 300,
                            temperature = 0.5,
                            top_p = 0.9,
                            top_k = 10,
                            is_greedy = False):
        
        if not self.is_load_model:
            self.load_model()

        dataset = ImageDataset(prompt = prompt,             
                               paths = [image],             
                               tokenizer = self.tokenizer,             
                               image_token_id = self.image_token_id,             
                               image_seq_length = self.image_seq_length)
        dataloader = DataLoader(dataset, 
                                collate_fn=dataset.collate_fn, 
                                num_workers=16, 
                                shuffle=False, 
                                drop_last=False, 
                                batch_size=batch_size)
        for batch in dataloader:
            first_batch = batch
            break
        
        captions = self.generate_base(first_batch, max_tokens, temperature, top_k, top_p, is_greedy)
        caption_str = prepend_string + captions[0] + append_string
        logger_i18n.debug("Caption: $$caption$$", {"$$caption$$": caption_str})
        return caption_str
    
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