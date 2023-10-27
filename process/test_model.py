import os
from PIL import Image
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers.utils import is_accelerate_available

HOME = os.getcwd()
PRETRAINED_MODEL_PATH = os.path.join(HOME, 'models')


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


# Set up the pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    PRETRAINED_MODEL_PATH,
    scheduler=DPMSolverMultistepScheduler.from_pretrained(
        PRETRAINED_MODEL_PATH, subfolder="scheduler"),
    torch_dtype=torch.float16,
).to("cuda")


def generate_image_grid(prompt,
                        num_samples=2,
                        num_rows=1):

    all_images = []
    for _ in range(num_rows):
        images = pipe([prompt] * num_samples, num_inference_steps=30,
                      guidance_scale=7.5).images
        all_images.extend(images)

    grid = image_grid(all_images, num_rows, num_samples)

    return grid


def main():
    prompt = "people in russia in the bright square 8k hiqh quality in the style <custom-art>"
    grid = generate_image_grid(prompt)
    grid.save('test.jpg')


if __name__ == "__main__":
    main()
