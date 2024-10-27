

from pathlib import Path
import torch
import torch.amp
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModel,
    AutoProcessor,
    AutoModelForCausalLM
)


from ._class.dataset import Prompt

from .utils import *
from .config import *


from accelerate import Accelerator
accelerator = Accelerator()

SCRIPT_DIR = Path(__file__).parent


# 图像筛选
image_paths = find_images_in_directory(Path(DATASET_PIC_ORIGIN_DIR), recursive=IMAGE_LOAD_RECURSIVE_BOOL)

logging_with_none_image(image_paths, raise_error=False)

if not CAPTION_OVERWRITE_BOOL:
    image_paths = filter_images_with_txt(image_paths)


# 加载模型
## tokenizer
tokenizer = AutoTokenizer.from_pretrained(JOY2_MODEL_STR, use_fast=True)

is_valid_tokenizer = isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast)
tokenizer_valid_error_info = i18n("Tokenizer is of type $$tokenizer_type$$", {"$$tokenizer_type$$": type(tokenizer)})

assert is_valid_tokenizer, tokenizer_valid_error_info

## MODEL
if JOY2_IS_NF4_BOOL: 
    from transformers import BitsAndBytesConfig
    nf4_config = BitsAndBytesConfig(load_in_4bit=True, 
                                    bnb_4bit_quant_type="nf4", 
                                    bnb_4bit_quant_storage=torch.bfloat16,
                                    bnb_4bit_use_double_quant=True, 
                                    bnb_4bit_compute_dtype=torch.bfloat16)

    llava_model = LlavaForConditionalGeneration.from_pretrained(JOY2_MODEL_STR, 
                                                                quantization_config=nf4_config, torch_dtype="bfloat16", 
                                                                device_map=device)
else: 
    llava_model = LlavaForConditionalGeneration.from_pretrained(JOY2_MODEL_STR, 
                                                                torch_dtype="bfloat16", 
                                                                device_map=device)

assert isinstance(llava_model, LlavaForConditionalGeneration)

dataset = ImageDataset(prompts, image_paths, tokenizer, llava_model.config.image_token_index, llava_model.config.image_seq_length)

dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=args.num_workers, shuffle=False, drop_last=False, batch_size=args.batch_size)

end_of_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
end_of_turn_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
assert isinstance(end_of_header_id, int) and isinstance(end_of_turn_id, int)

pbar = tqdm(total=len(image_paths), desc="Captioning images...", dynamic_ncols=True)

for batch in dataloader:
    vision_dtype = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
    vision_device = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
    language_device = llava_model.language_model.get_input_embeddings().weight.device

    # Move to GPU
    pixel_values = batch['pixel_values'].to(vision_device, non_blocking=True)
    input_ids = batch['input_ids'].to(language_device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(language_device, non_blocking=True)

    # Normalize the image
    pixel_values = pixel_values / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
    pixel_values = pixel_values.to(vision_dtype)

    # Generate the captions
    generate_ids = llava_model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        suppress_tokens=None,
        use_cache=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Trim off the prompts
    assert isinstance(generate_ids, torch.Tensor)
    generate_ids = generate_ids.tolist()
    generate_ids = [trim_off_prompt(ids, end_of_header_id, end_of_turn_id) for ids in generate_ids]

    # Decode the captions
    captions = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    captions = [c.strip() for c in captions]

    for path, caption in zip(batch['paths'], captions):
        write_caption(Path(path), caption)
    
    pbar.update(len(captions))