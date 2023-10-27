import os
import math
from itertools import chain
from accelerate.logging import get_logger
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from accelerate import Accelerator


class Trainer():
    def __init__(self,
                 train_dataset,
                 save_model_path,
                 hyperparameters,
                 placeholder_token,
                 initializer_token,
                 pretrained_model_name):

        self.train_dataset = train_dataset

        self.hyperparameters = hyperparameters
        self.save_model_path = save_model_path
        self.pretrained_model_name = pretrained_model_name

        self.placeholder_token = placeholder_token
        self.initializer_token = initializer_token,
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name,
            subfolder="tokenizer")

        self.__num_added_tokens = None
        self.__token_ids = None

        self.initializer_token_id = self.token_ids[0]
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            placeholder_token)

        # Load models and create wrapper for stable diffusion
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name, subfolder="unet")

        self.logger = get_logger(__name__)
        self.noise_scheduler = DDPMScheduler.from_config(
            pretrained_model_name, subfolder="scheduler")

        self.initialize_token_embeddings()
        self.freeze_models_params()

    @property
    def num_added_tokens(self):
        # Add the placeholder token in tokenizer
        if self.__num_added_tokens is None:
            added_tokens = self.tokenizer.add_tokens(self.placeholder_token)
            if added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {self.placeholder_token}. Please pass a different"
                    " `placeholder_token` that is not already in the tokenizer."
                )
            self.__num_added_tokens = added_tokens

        return self.__num_added_tokens

    @property
    def token_ids(self):
        # Convert the initializer_token, placeholder_token to ids
        if self.__token_ids is None:
            ids = self.tokenizer.encode(
                self.initializer_token, add_special_tokens=False)
            if len(ids) > 1:
                raise ValueError(
                    "The initializer token must be a single token.")
            self.__token_ids = ids

        return self.__token_ids

    def initialize_token_embeddings(self):
        # Resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        self.token_embeds = self.text_encoder.get_input_embeddings().weight.data
        self.token_embeds[self.placeholder_token_id] = self.token_embeds[self.initializer_token_id]

    @staticmethod
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    def freeze_models_params(self):
        # Freeze vae and unet
        Trainer.freeze_params(self.vae.parameters())
        Trainer.freeze_params(self.unet.parameters())
        # Freeze all parameters except for the token embeddings in text encoder
        text_encoder_params = chain(
            self.text_encoder.text_model.encoder.parameters(),
            self.text_encoder.text_model.final_layer_norm.parameters(),
            self.text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        Trainer.freeze_params(text_encoder_params)

    def create_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hyperparameters["train_batch_size"],
            shuffle=True)

        return dataloader

    def save_progress(self, accelerator, save_path):
        self.logger.info("Saving embeddings")
        learned_embeds = accelerator.unwrap_model(
            self.text_encoder).get_input_embeddings().weight[self.placeholder_token_id]
        learned_embeds_dict = {
            self.placeholder_token: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, os.path.join(
            self.save_model_path, save_path))

    def train(self):
        train_batch_size = self.hyperparameters["train_batch_size"]
        gradient_accumulation_steps = self.hyperparameters["gradient_accumulation_steps"]
        learning_rate = self.hyperparameters["learning_rate"]
        max_train_steps = self.hyperparameters["max_train_steps"]
        gradient_checkpointing = self.hyperparameters["gradient_checkpointing"]

        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=self.hyperparameters["mixed_precision"]
        )

        if gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

        train_dataloader = self.create_dataloader()

        if self.hyperparameters["scale_lr"]:
            learning_rate = (
                learning_rate * gradient_accumulation_steps *
                train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
            lr=learning_rate,
        )

        text_encoder, optimizer, train_dataloader = accelerator.prepare(
            text_encoder, optimizer, train_dataloader
        )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and unet to device
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.unet.to(accelerator.device, dtype=weight_dtype)

        # Keep vae in eval mode as we don't train it
        self.vae.eval()
        # Keep unet in train mode to enable gradient checkpointing
        self.unet.train()

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / gradient_accumulation_steps)
        num_train_epochs = math.ceil(
            max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = train_batch_size * \
            accelerator.num_processes * gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(
            f"  Instantaneous batch size per device = {train_batch_size}")
        self.logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(
            f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(max_train_steps),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(num_train_epochs):
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    latents = self.vae.encode(batch["pixel_values"].to(
                        dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = self.unet(
                        noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(
                            latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(noise_pred, target, reduction="none").mean(
                        [1, 2, 3]).mean()
                    accelerator.backward(loss)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(
                        len(self.tokenizer)) != self.placeholder_token_id
                    grads.data[index_grads_to_zero,
                               :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % self.hyperparameters["save_steps"] == 0:
                        save_path = f"learned_embeds-step-{global_step}.bin"
                        self.save_progress(accelerator, save_path)

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.pretrained_model_name,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=self.tokenizer,
                vae=self.vae,
                unet=self.unet,
            )
            pipeline.save_pretrained(self.save_model_path)
            # Also save the newly trained embeddings
            save_path = f"learned_embeds.bin"
            self.save_progress(accelerator, save_path)
