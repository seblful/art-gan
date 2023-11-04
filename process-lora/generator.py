import os
from PIL import Image
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers.utils import is_accelerate_available


class ImagesGenerator():
    def __init__(self,
                 pretrained_model_name_or_path,
                 lora_model_path,
                 prompt,
                 negative_prompt,
                 scale,
                 guidance_scale,
                 num_gen_images):

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lora_model_path = lora_model_path
        self.prompt = prompt.strip()
        self.negative_prompt = negative_prompt.strip()
        self.scale = scale
        self.guidance_scale = guidance_scale
        self.num_gen_images = num_gen_images

        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.__pipeline = None

        self.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config)

    @property
    def pipeline(self):
        if self.__pipeline is None:
            self.__pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                torch_dtype=torch.float16, use_safetensors=False)
            self.__pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.__pipeline.scheduler.config)
            self.__pipeline.unet.load_attn_procs(self.lora_model_path)
            self.__pipeline.to(self.device)

        return self.__pipeline

    def generate_list_images(self):
        images = self.pipeline(prompt=[self.prompt] * self.num_gen_images,
                               negative_prompt=[
                                   self.negative_prompt] * self.num_gen_images,
                               num_inference_steps=25,
                               guidance_scale=self.guidance_scale,
                               cross_attention_kwargs={"scale": self.scale}).images

        return images
