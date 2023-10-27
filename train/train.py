from trainer import Trainer
from dataset import TextualInversionDataset

import os

HOME = os.getcwd()
IMAGES_PATH = os.path.join(HOME, 'images')
SAVE_MODEL_PATH = os.path.join(HOME, 'runs')

PRETRAINED_MODEL_NAME = "stabilityai/stable-diffusion-2-1"

WHAT_TO_TEACH = "style"
PLACEHOLDER_TOKEN = "custom-art"
INITIALIZER_TOKEN = "art"

# Setting up all training args
HYPERPARAMETERS = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": 2000,
    "save_steps": 250,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42
}


def main():
    train_dataset = TextualInversionDataset(
        data_root=IMAGES_PATH,
        size=vae.sample_size,
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        learnable_property=WHAT_TO_TEACH,
        placeholder_token=PLACEHOLDER_TOKEN,
        repeats=100,
        # Option selected above between object and style
        center_crop=False,
        set="train",
    )

    trainer = Trainer(train_dataset=train_dataset,
                      save_model_path=SAVE_MODEL_PATH,
                      hyperparameters=HYPERPARAMETERS,
                      placeholder_token=PLACEHOLDER_TOKEN,
                      initializer_token=INITIALIZER_TOKEN,
                      pretrained_model_name=PRETRAINED_MODEL_NAME)


if __name__ == "__main__":
    main()
