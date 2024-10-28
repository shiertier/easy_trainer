import dataclasses
from torch.utils.data import Dataset
from pathlib import Path
import torch
import torch.amp
import torchvision.transforms.functional as TVF
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from ..utils import logger_i18n

@dataclasses.dataclass
class Prompt:
    prompt: str
    weight: float

class ImageDataset(Dataset):
    def __init__(
        self,
        prompt: str,
        paths: list[Path],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        image_token_id: int,
        image_seq_length: int,
    ):
        self.prompt = prompt
        self.paths = paths
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.pad_token_id = tokenizer.pad_token_id  # 填充标记的ID

    def __len__(self):
        return len(self.paths)  # 返回数据集的长度

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]  # 获取图像路径

        try:
            image = Image.open(path)
            if image.size != (384, 384):
                image = image.resize((384, 384), Image.LANCZOS)
            image = image.convert("RGB")  # 转换为RGB格式
            pixel_values = TVF.pil_to_tensor(image)  # 将图像转换为张量
        except Exception as e:
            logger_i18n.error(
                "Failed to load image '$$path$$': $$e$$",
                {"$$path$$": path, "$$e$$": e},
            )
            pixel_values = None  # 后续将被过滤掉

        # 构建对话内容
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": self.prompt,
            },
        ]

        # 格式化对话内容
        convo_string = self.tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(convo_string, str)

        # 对对话进行分词
        convo_tokens = self.tokenizer.encode(
            convo_string, add_special_tokens=False, truncation=False
        )

        # 根据图像序列长度重复图像标记
        input_tokens = []
        for token in convo_tokens:
            if token == self.image_token_id:
                input_tokens.extend([self.image_token_id] * self.image_seq_length)  # 扩展图像标记
            else:
                input_tokens.append(token)

        input_ids = torch.tensor(input_tokens, dtype=torch.long)  # 转换为张量
        attention_mask = torch.ones_like(input_ids)  # 创建注意力掩码

        return {
            "path": path,  # 返回路径
            "pixel_values": pixel_values,  # 返回像素值
            "input_ids": input_ids,  # 返回输入ID
            "attention_mask": attention_mask,  # 返回注意力掩码
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        # 过滤掉未成功加载的图像
        batch = [item for item in batch if item["pixel_values"] is not None]

        if not batch:
            return {
                "paths": [],
                "pixel_values": torch.empty(0),  # 空的像素值张量
                "input_ids": torch.empty(0),  # 空的输入ID张量
                "attention_mask": torch.empty(0),  # 空的注意力掩码张量
            }

        # 填充 input_ids 和 attention_mask
        max_length = max(item["input_ids"].shape[0] for item in batch)
        n_pad = [max_length - item["input_ids"].shape[0] for item in batch]
        input_ids = torch.stack(
            [
                torch.nn.functional.pad(item["input_ids"], (n, 0), value=self.pad_token_id)
                for item, n in zip(batch, n_pad)
            ]
        )
        attention_mask = torch.stack(
            [
                torch.nn.functional.pad(item["attention_mask"], (n, 0), value=0)
                for item, n in zip(batch, n_pad)
            ]
        )

        # 堆叠像素值
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        # 图像路径
        paths = [item["path"] for item in batch]

        return {
            "paths": paths,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
