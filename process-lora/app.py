from generator import ImagesGenerator

import os
import numpy as np
import gradio as gr

HOME = os.getcwd()
BASE_MODEL_NAME = "stabilityai/stable-diffusion-2-1"
LORA_MODEL_PATH = os.path.join(HOME, "checkpoint")


# prompt = "people in russia in the bright square 8k high quality in the style <mvQypyI9Sqnnwve1FH>"


# images_generator = ImagesGenerator(
#     pretrained_model_path=PRETRAINED_MODEL_PATH)


def generate_images(prompt,
                    negative_prompt,
                    scale,
                    guidance_scale,
                    num_gen_images=4):

    images_generator = ImagesGenerator(pretrained_model_name_or_path=BASE_MODEL_NAME,
                                       lora_model_path=LORA_MODEL_PATH,
                                       prompt=prompt,
                                       negative_prompt=negative_prompt,
                                       scale=scale,
                                       guidance_scale=guidance_scale,
                                       num_gen_images=num_gen_images)

    gen_images = images_generator.generate_list_images()

    return gen_images


# def generate_images(user_prompt, user_negative_prompt):
#     print(type(user_prompt), user_prompt)
#     print(type(user_negative_prompt), user_negative_prompt,
#           user_negative_prompt if user_negative_prompt.strip() else None)
#     # Generate four images using stable diffusion
#     images = []
#     for _ in range(4):
#         # Replace this code with your own stable diffusion implementation
#         # This is just a placeholder code that generates random noise
#         image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
#         images.append(image)
#     return images


def gradio_app():
    with gr.Blocks(title="Art Image Generator", theme='base') as combined_sd:
        with gr.Row():
            with gr.Column():
                image_generation_header = gr.Markdown('''
                # Stable Diffusion Image Generation
                                                      
                Enter a prompt for what you would like Stable Diffusion to generate and click the "Generate Image" button to watch the result.
                ''')

                user_prompt = gr.Textbox(placeholder="Enter your prompt")
                user_negative_prompt = gr.Textbox(
                    placeholder="Enter your negative prompt")

                generate_image_button = gr.Button("Generate Image")

                scale_slider = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.05, label="Scale")
                guidance_scale_slider = gr.Slider(
                    minimum=0, maximum=30, value=7.5, label="Guidance Scale")

            with gr.Column():
                output_gallery = gr.Gallery(label="Generated Images")

        generate_image_button.click(fn=generate_images,
                                    inputs=[user_prompt,
                                            user_negative_prompt,
                                            scale_slider,
                                            guidance_scale_slider],
                                    outputs=[output_gallery])

    combined_sd.launch()


def main():
    gradio_app()


if __name__ == "__main__":
    main()
