import re
import os
import gc
import torchvision.transforms.functional as TVF
import torch.distributed
import torch
import torch.amp.autocast_mode
import json
from torch.utils.data import DataLoader
from torch import nn
from accelerate import Accelerator
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
from PIL import Image
from tqdm import tqdm

accelerator = Accelerator()

HF_TOKEN = os.environ.get("HF_TOKEN", None)

HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", None)
if HF_CACHE_DIR is None:
    HF_CACHE_DIR = '/root/.cache/huggingface'

CLIP_PATH = "google/siglip-so400m-patch14-384"
CHECKPOINT_PATH = Path("cgrkzexw-599808")

print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}] Basic Info:")
print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}]     Total processes: {accelerator.num_processes}")
print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}]     HF_CACHE_DIR: {HF_CACHE_DIR}")
print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}]     CLIP_PATH: {CLIP_PATH}")
print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}]     CHECKPOINT_PATH: {CHECKPOINT_PATH}")
print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}]     HF_TOKEN: {HF_TOKEN}")


TITLE = "<h1><center>JoyCaption Alpha Two (2024-09-26a)</center></h1>"
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

HF_TOKEN = os.environ.get("HF_TOKEN", None)


class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)   # Matches HF's implementation of llama3

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # <|image_start|>, IMAGE, <|image_end|>
        other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        assert other_tokens.shape == (x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)


# Load CLIP Model
accelerator.print("Loading CLIP")
clip_processor = AutoProcessor.from_pretrained(CLIP_PATH, cache_dir=HF_CACHE_DIR)
clip_model = AutoModel.from_pretrained(CLIP_PATH, cache_dir=HF_CACHE_DIR)
clip_model = clip_model.vision_model

assert (CHECKPOINT_PATH / "clip_model.pt").exists()
accelerator.print("Loading VLM's custom vision model")
checkpoint = torch.load(CHECKPOINT_PATH / "clip_model.pt", map_location='cpu')
checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
clip_model.load_state_dict(checkpoint)
del checkpoint

# Compile CLIP Model
clip_model.eval()
clip_model.requires_grad_(False)
clip_model.to(accelerator.device)
clip_model = torch.compile(clip_model, backend='inductor', mode='default')

# Load Tokenizer
accelerator.print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH / "text_model", use_fast=True, cache_dir=HF_CACHE_DIR)
assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

# Load and Compile Text Model
accelerator.print("Loading LLM")
accelerator.print("Loading VLM's custom text model")
text_model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH / "text_model",
    torch_dtype=torch.bfloat16,
    cache_dir=HF_CACHE_DIR
)
text_model = text_model.to(accelerator.device)
text_model = torch.compile(text_model, backend='inductor', mode='default')
text_model.eval()

# Load and Compile Image Adapter
accelerator.print("Loading image adapter")
image_adapter = ImageAdapter(
    input_features=clip_model.config.hidden_size,
    output_features=text_model.config.hidden_size,
    ln1=False,
    pos_emb=False,
    num_image_tokens=38,
    deep_extract=False
)
image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location='cpu'))
image_adapter = image_adapter.to(accelerator.device)
image_adapter = torch.compile(image_adapter, backend='inductor', mode='default')
image_adapter.eval()


def deduplicate(tags):
    res = []
    for tag in tags:
        if tag not in res:
            res.append(tag)
    return res


@torch.no_grad()
def stream_chat(input_image: Image.Image, caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str, custom_prompt: str) -> tuple[str, str]:
    # torch.cuda.empty_cache()

    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()
    else:

        # 'any' means no length specified
        length = None if caption_length == "any" else caption_length

        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        # Build prompt
        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")

        prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

        # Add extra options
        if len(extra_options) > 0:
            prompt_str += " " + " ".join(extra_options)

        # Add name, length, word_count
        prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    # For debugging
    # logger.accelerator.print(f"Prompt: {logging.yellow(prompt_str)}")

    # Preprocess image
    # NOTE: I found the default processor for so400M to have worse results than just using PIL directly
    # image = clip_processor(images=input_image, return_tensors='pt').pixel_values
    image = input_image.resize((384, 384), Image.LANCZOS)
    pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])

    pixel_values = pixel_values.to(accelerator.device)

    # Embed image
    # This results in Batch x Image Tokens x Features
    with torch.amp.autocast_mode.autocast(str(accelerator.device), enabled=True):
        vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
        embedded_images = image_adapter(vision_outputs.hidden_states)
        embedded_images = embedded_images.to(accelerator.device)

    # Build the conversation
    convo = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": prompt_str,
        },
    ]

    # Format the conversation
    convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    assert isinstance(convo_string, str)

    # Tokenize the conversation
    # prompt_str is tokenized separately so we can do the calculations below
    convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
    prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)
    assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
    convo_tokens = convo_tokens.squeeze(0)   # Squeeze just to make the following easier
    prompt_tokens = prompt_tokens.squeeze(0)

    # Calculate where to inject the image
    eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[0].tolist()
    assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

    preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]   # Number of tokens before the prompt

    # Embed the tokens
    convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(accelerator.device))

    # Construct the input
    # print(f"devices | convo_embeds: {convo_embeds.device} | embedded_images: {embedded_images.device}")
    input_embeds = torch.cat([
        convo_embeds[:, :preamble_len],   # Part before the prompt
        embedded_images.to(dtype=convo_embeds.dtype),   # Image
        convo_embeds[:, preamble_len:],   # The prompt and anything after it
    ], dim=1).to(accelerator.device)

    input_ids = torch.cat([
        convo_tokens[:preamble_len].unsqueeze(0),
        # Dummy tokens for the image (TODO: Should probably use a special token here so as not to confuse any generation algorithms that might be inspecting the input)
        torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
        convo_tokens[preamble_len:].unsqueeze(0),
    ], dim=1).to(accelerator.device)
    attention_mask = torch.ones_like(input_ids)

    # Debugging
    # accelerator.print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

    # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=False, suppress_tokens=None)
    # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5, suppress_tokens=None)
    generate_ids = text_model.generate(input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask, max_new_tokens=300,
                                       do_sample=True, suppress_tokens=None)   # Uses the default which is temp=0.6, top_p=0.9

    # Trim off the prompt
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>"):
        generate_ids = generate_ids[:, :-1]

    caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    if caption and caption[-1] not in ('.', '!', '?', '\"', '\'') and len(sentences := caption.split('.')) > 1:
        caption = '. '.join(sentences[:-1]) + '.'  # Remove the last sentence if it's incomplete.

    # deduplicate
    caption = ', '.join(deduplicate(caption.split(', ')))

    return prompt_str, caption.strip()


def load_images_from_directory(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths


if __name__ == "__main__":
    target_directory = "/image/207764205"  # Replace with your target directory
    save_path = "/image/joy_caption_results.json"
    debug = False
    resume = False

    prompt = '''Write a stable-diffusion prompt within 150 words for this image.
Do NOT use any ambiguous language.
ONLY include content you are absolutely sure of. 
'''

    save_interval = 10
    clean_cache_interval = 100
    dataloader_max_workers = 0

    image_paths = load_images_from_directory(target_directory)
    global_n_imgs = len(image_paths)
    image_paths = [image_paths[i] for i in range(accelerator.local_process_index, len(image_paths), accelerator.num_processes)]
    local_n_imgs = len(image_paths)
    dataloader = DataLoader(image_paths, batch_size=1, shuffle=False, num_workers=dataloader_max_workers)

    if accelerator.num_processes > 1:
        local_save_path = os.path.splitext(save_path)[0] + f"_{accelerator.local_process_index}" + os.path.splitext(save_path)[1]
    else:
        local_save_path = save_path

    if resume and os.path.exists(local_save_path):
        with open(local_save_path, "r", encoding="utf-8") as f:
            joy_captions = json.load(f)
        print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}] Resumed {len(joy_captions)} results from {local_save_path}")
    else:
        joy_captions = {}

    print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}] Process Started!")
    print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}]     Process images: {local_n_imgs}/{global_n_imgs}")
    print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}]     Save interval: {save_interval}")
    print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}]     Clean cache interval: {clean_cache_interval}")
    print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}]     Dataloader max workers: {dataloader_max_workers}")

    # dataset = FastDataset(dataset_src)  # This is just used to load dataset conveniently
    step = 0
    for batch in tqdm(dataloader, desc=f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}]", position=accelerator.local_process_index):
        img_path = batch[0]
        img_key = os.path.splitext(os.path.basename(img_path))[0]
        if img_key in joy_captions:
            continue

        img = Image.open(img_path)

        # Assuming 'caption' is something you want to generate based on the image
        # If you have metadata in a separate file or in the filename, you would extract it here
        caption = None  # Placeholder for the caption logic

        pmt = prompt.format(caption) if caption else prompt.format("")
        if debug:
            tqdm.write(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}] Prompt: {pmt}")
        _, caption = stream_chat(
            input_image=img,
            caption_type=None,
            caption_length=None,
            extra_options=None,
            name_input=None,
            custom_prompt=pmt,
        )
        if debug:
            tqdm.write(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}] Caption: {caption}")
        if not caption:
            if debug:
                tqdm.write(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}] Empty caption for {img_key}")
        else:
            joy_captions[img_key] = caption
        step += 1

        if step % save_interval == 0:
            with open(local_save_path, "w", encoding="utf-8") as f:
                json.dump(joy_captions, f, ensure_ascii=False, indent=4)
            tqdm.write(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}] Saved {step} results to {local_save_path}")

        if step % clean_cache_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()

    with open(local_save_path, "w", encoding="utf-8") as f:
        json.dump(joy_captions, f, ensure_ascii=False, indent=4)

    print(f"[Process {accelerator.local_process_index+1}/{accelerator.num_processes}] Process done! Saved {step} results to {local_save_path}. Waiting for others to finish...")
    accelerator.wait_for_everyone()