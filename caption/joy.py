

from pathlib import Path
import torch
import torch.amp
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

SCRIPT_DIR = Path(__file__).parent

class Joy2:
    def __init__(self, 
                 model_name, 
                 devices_key = "all",
                 is_nf4 = True,
                 local = True,

                 ):
        self.model_name = model_name
        self.devices = convert_to_device_map(devices_key)
        self._is_nf4 = is_nf4
        self.is_load_model = False
        if local:
            self.load_model()

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        is_valid_tokenizer = isinstance(self.tokenizer, PreTrainedTokenizer) or isinstance(self.tokenizer, PreTrainedTokenizerFast)
        tokenizer_valid_error_info = i18n("Tokenizer is of type $$tokenizer_type$$", {"$$tokenizer_type$$": type(self.tokenizer)})
        assert is_valid_tokenizer, tokenizer_valid_error_info

        self.end_of_header_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        self.end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        assert isinstance(self.end_of_header_id, int) and isinstance(self.end_of_turn_id, int)

    def load_model(self):
        if self.is_load_model:
            return
        self.load_tokenizer()
        if self._is_nf4: 
            from transformers import BitsAndBytesConfig
            nf4_config = BitsAndBytesConfig(load_in_4bit=True, 
                                            bnb_4bit_quant_type="nf4", 
                                            bnb_4bit_quant_storage=torch.bfloat16,
                                            bnb_4bit_use_double_quant=True, 
                                            bnb_4bit_compute_dtype=torch.bfloat16)

            self.llava_model = LlavaForConditionalGeneration.from_pretrained(JOY2_MODEL_STR, 
                                                                             quantization_config=nf4_config, 
                                                                             torch_dtype="bfloat16")
        else: 
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(JOY2_MODEL_STR, 
                                                                        torch_dtype="bfloat16",)

        if is_single_gpu(self.devices):
            self.llava_model.to(self.devices)
        else:
            from accelerate import Accelerator
            set_env("CUDA_VISIBLE_DEVICES", self.devices)
            accelerator = Accelerator()
            self.llava_model.to(accelerator.device)

        assert isinstance(self.llava_model, LlavaForConditionalGeneration)

        self.vision_dtype = self.llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        self.vision_device = self.llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
        self.language_device = self.llava_model.language_model.get_input_embeddings().weight.device
        self.image_token_id = self.llava_model.config.image_token_index
        self.image_seq_length = self.llava_model.config.image_seq_length

        self.is_load_model = True

    def generate_base(self, dataloader, max_tokens, temperature, top_k, top_p, is_greedy):
        pixel_values = dataloader['pixel_values'].to(self.vision_device, non_blocking=True)
        input_ids = dataloader['input_ids'].to(self.language_device, non_blocking=True)
        attention_mask = dataloader['attention_mask'].to(self.language_device, non_blocking=True)

        # Normalize the image
        pixel_values = pixel_values / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(self.vision_dtype)

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

    def generate_captions( self, 
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

    def generate_caption(   self, 
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
        
        captions = self.generate_base(dataloader, max_tokens, temperature, top_k, top_p, is_greedy)
        caption_str = prepend_string + captions[0] + append_string
        return caption_str
    
    def create_service(self, server_name, server_port):
        # TODO
        pass

model_name = JOY2_MODEL_STR
devices_key = JOY2_GPU_COUNT_STR
is_nf4 = JOY2_IS_NF4_BOOL
local = True
image_dir = DATASET_PIC_ORIGIN_DIR
is_image_load_recursive = IMAGE_LOAD_RECURSIVE_BOOL
overwrite_caption = CAPTION_OVERWRITE_BOOL
batch_size = JOY2_MAX_BATCH_SIZE
max_tokens = JOY2_MAX_NEW_TOKENS_INT
prepend_string = JOY2_ADD_PREPEND_STR
append_string = JOY2_ADD_APPEND_STR
is_greedy = JOY2_IS_GREEDY_BOOL