import os
from PIL import Image
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers.utils import is_accelerate_available


class ImagesGenerator():
    def __init__(self,
                 pretrained_model_path,
                 prompt,
                 negative_prompt,
                 num_gen_images):

        self.pretrained_model_path = pretrained_model_path
        self.prompt = prompt.strip()
        self.negative_prompt = negative_prompt.strip()
        self.num_gen_images = num_gen_images

        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            self.pretrained_model_path, subfolder="scheduler")

        self.__pipeline = None

    @property
    def pipeline(self):
        if self.__pipeline is None:
            self.__pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_path,
                scheduler=self.scheduler,
                torch_dtype=torch.float16,
            ).to(self.device)

        return self.__pipeline

    def generate_list_images(self):
        images = self.pipeline(prompt=[self.prompt] * self.num_gen_images,
                               negative_prompt=[
                                   self.negative_prompt] * self.num_gen_images,
                               num_inference_steps=30,
                               guidance_scale=7.5).images

        return images
