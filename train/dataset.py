import os
import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        pretrained_model_name,
        learnable_property="style",
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name,
            subfolder="tokenizer")
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(
            self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": Image.BILINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]

        self.__templates = None
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    @property
    def templates(self):
        if self.__templates is None:
            self.__templates = [
                "a painting in the style of {}",
                "a rendering in the style of {}",
                "a cropped painting in the style of {}",
                "the painting in the style of {}",
                "a clean painting in the style of {}",
                "a dirty painting in the style of {}",
                "a dark painting in the style of {}",
                "a picture in the style of {}",
                "a cool painting in the style of {}",
                "a close-up painting in the style of {}",
                "a bright painting in the style of {}",
                "a cropped painting in the style of {}",
                "a good painting in the style of {}",
                "a close-up painting in the style of {}",
                "a rendition in the style of {}",
                "a nice painting in the style of {}",
                "a small painting in the style of {}",
                "a weird painting in the style of {}",
                "a large painting in the style of {}",
            ]
        return self.__templates

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2,
                      (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size),
                             resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
