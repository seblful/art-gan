import os
import requests
import toml
import glob
from accelerate.utils import write_basic_config
from tqdm import tqdm

PROJECT_NAME = "last"

HOME = os.getcwd()
TRAIN_DATA_DIR = os.path.join(HOME, 'images')
LORA_DIR = os.path.join("LoRA")
BASE_MODELS_DIR = os.path.join(HOME, "base-models")
OUTPUT_DIR = os.path.join(LORA_DIR, "output")
CONFIG_DIR = os.path.join(LORA_DIR, "config")
LOGGING_DIR = os.path.join(LORA_DIR, "logs")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "sample")

CONFIG_PATH = os.path.join(CONFIG_DIR, "config_file.toml")
DATASET_CONFIG_PATH = os.path.join(CONFIG_DIR, "dataset_config.toml")
ACCELERATE_CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")
SAMPLE_PROMPT_PATH = os.path.join(CONFIG_DIR, "sample_prompt.txt")

MODEL_PATH = os.path.join(
    BASE_MODELS_DIR, "stable-diffusion-2-1-base.safetensors")
VAE_PATH = os.path.join(BASE_MODELS_DIR, "stablediffusion.vae.pt")
NETWORK_WEIGHT_PATH = os.path.join(BASE_MODELS_DIR, "art.safetensors")


HF_TOKEN = "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"
MODEL_NAME = "stable-diffusion-2-1-base"
MODEL_URL = "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors"
VAE_URL = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt"


v2 = True
v_parameterization = False

dataset_repeats = 10
activation_word = "mksks"
caption_extension = ".txt"
resolution = 512
flip_aug = False
keep_tokens = 0


def create_dirs(list_of_dirs):
    for dir in list_of_dirs:
        os.makedirs(dir, exist_ok=True)


def download_file(url, filename, chunk_size=8192):
    if not os.path.exists(filename):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=chunk_size), desc=f"Downloading file {filename}"):
                    f.write(chunk)

    else:
        print(f"Base model with filename {filename} was existed before.")

    return filename


def parse_folder_name(folder_name, default_num_repeats, default_class_token):
    folder_name_parts = folder_name.split("_")

    if len(folder_name_parts) == 2:
        if folder_name_parts[0].isdigit():
            num_repeats = int(folder_name_parts[0])
            class_token = folder_name_parts[1].replace("_", " ")
        else:
            num_repeats = default_num_repeats
            class_token = default_class_token
    else:
        num_repeats = default_num_repeats
        class_token = default_class_token

    return num_repeats, class_token


def find_image_files(path):
    supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    return [file for file in glob.glob(path + '/**/*', recursive=True) if file.lower().endswith(supported_extensions)]


def process_data_dir(data_dir, default_num_repeats, default_class_token, is_reg=False):
    subsets = []

    images = find_image_files(data_dir)
    if images:
        subsets.append({
            "image_dir": data_dir,
            "class_tokens": default_class_token,
            "num_repeats": default_num_repeats,
            **({"is_reg": is_reg} if is_reg else {}),
        })

    for root, dirs, files in os.walk(data_dir):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            images = find_image_files(folder_path)

            if images:
                num_repeats, class_token = parse_folder_name(
                    folder, default_num_repeats, default_class_token)

                subset = {
                    "image_dir": folder_path,
                    "class_tokens": class_token,
                    "num_repeats": num_repeats,
                }

                if is_reg:
                    subset["is_reg"] = True

                subsets.append(subset)

    return subsets


def train(config):
    args = ""
    for k, v in config.items():
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'--{k}="{v}" '
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    return args


def main():

    create_dirs([LORA_DIR, BASE_MODELS_DIR, OUTPUT_DIR,
                CONFIG_DIR, LOGGING_DIR, SAMPLE_DIR])

    for url, model_name in zip((MODEL_URL, VAE_URL), (MODEL_PATH, VAE_PATH)):
        download_file(url, model_name)

    write_basic_config(save_location=ACCELERATE_CONFIG_PATH)

    subsets = process_data_dir(
        TRAIN_DATA_DIR, dataset_repeats, activation_word)

    config = {
        "general": {
            "enable_bucket": True,
            "caption_extension": caption_extension,
            "shuffle_caption": True,
            "keep_tokens": keep_tokens,
            "bucket_reso_steps": 64,
            "bucket_no_upscale": False,
        },
        "datasets": [
            {
                "resolution": resolution,
                "min_bucket_reso": 320 if resolution > 640 else 256,
                "max_bucket_reso": 1280 if resolution > 640 else 1024,
                "caption_dropout_rate": 0,
                "caption_tag_dropout_rate": 0,
                "caption_dropout_every_n_epochs": 0,
                "flip_aug": flip_aug,
                "color_aug": False,
                "face_crop_aug_range": None,
                "subsets": subsets,
            }
        ],
    }

    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None

    config_str = toml.dumps(config)

    with open(DATASET_CONFIG_PATH, "w") as f:
        f.write(config_str)

    # print(config_str)

    network_category = "LoRA"
    conv_dim = 32
    conv_alpha = 16
    network_dim = 32
    network_alpha = 16
    network_weight = ""
    network_module = "lycoris.kohya" if network_category in [
        "LoHa", "LoCon_Lycoris"] else "networks.lora"
    network_args = "" if network_category == "LoRA" else [
        f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}",
    ]

    min_snr_gamma = -1
    # ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
    optimizer_type = "AdamW8bit"
    optimizer_args = ""
    train_unet = True
    unet_lr = 1e-4
    train_text_encoder = True
    text_encoder_lr = 5e-5

    # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
    lr_scheduler = "constant"
    lr_warmup_steps = 0
    lr_scheduler_num_cycles = 0
    lr_scheduler_power = 0

    if not network_weight:
        print("  - No LoRA weight loaded.")
    else:
        if os.path.exists(network_weight):
            print(f"  - Loading LoRA weight: {network_weight}")
        else:
            print(f"  - {network_weight} does not exist.")
            network_weight = ""

    print("- Optimizer Config:")
    print(f"  - Additional network category: {network_category}")
    print(f"  - Using {optimizer_type} as Optimizer")
    if optimizer_args:
        print(f"  - Optimizer Args: {optimizer_args}")
    if train_unet and train_text_encoder:
        print("  - Train UNet and Text Encoder")
        print(f"    - UNet learning rate: {unet_lr}")
        print(f"    - Text encoder learning rate: {text_encoder_lr}")
    if train_unet and not train_text_encoder:
        print("  - Train UNet only")
        print(f"    - UNet learning rate: {unet_lr}")
    if train_text_encoder and not train_unet:
        print("  - Train Text Encoder only")
        print(f"    - Text encoder learning rate: {text_encoder_lr}")
    print(f"  - Learning rate warmup steps: {lr_warmup_steps}")
    print(f"  - Learning rate Scheduler: {lr_scheduler}")
    if lr_scheduler == "cosine_with_restarts":
        print(f"  - lr_scheduler_num_cycles: {lr_scheduler_num_cycles}")
    elif lr_scheduler == "polynomial":
        print(f"  - lr_scheduler_power: {lr_scheduler_power}")

    lowram = True
    enable_sample_prompt = True
    sampler = "ddim"
    noise_offset = 0.0
    num_epochs = 50
    vae_batch_size = 1
    train_batch_size = 1
    mixed_precision = "fp16"  # ["no","fp16","bf16"]
    save_precision = "fp16"  # ["float", "fp16", "bf16"]
    # ["save_every_n_epochs", "save_n_epoch_ratio"]
    save_n_epochs_type = "save_every_n_epochs"
    save_n_epochs_type_value = 1
    save_model_as = "safetensors"  # ["ckpt", "pt", "safetensors"]
    max_token_length = 225
    clip_skip = 2
    gradient_checkpointing = False
    gradient_accumulation_steps = 1
    seed = -1

    prior_loss_weight = 1.0

    sample_str = f"""
    image in high resolution 8k \
        --n lowres, text, error, low quality, low resolution, low details, jpeg artifacts, signature, watermark, blurry \
            --w 512 \
                --h 768 \
                    --l 7 \
                        --s 28
    """

    config = {
        "model_arguments": {
            "v2": v2,
            "v_parameterization": v_parameterization if v2 and v_parameterization else False,
            "pretrained_model_name_or_path": MODEL_PATH,
            "vae": VAE_PATH,
        },
        "additional_network_arguments": {
            "no_metadata": False,
            "unet_lr": float(unet_lr) if train_unet else None,
            "text_encoder_lr": float(text_encoder_lr) if train_text_encoder else None,
            "network_weights": network_weight,
            "network_module": network_module,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_args": network_args,
            "network_train_unet_only": True if train_unet and not train_text_encoder else False,
            "network_train_text_encoder_only": True if train_text_encoder and not train_unet else False,
            "training_comment": None,
        },
        "optimizer_arguments": {
            "min_snr_gamma": min_snr_gamma if not min_snr_gamma == -1 else None,
            "optimizer_type": optimizer_type,
            "learning_rate": unet_lr,
            "max_grad_norm": 1.0,
            "optimizer_args": eval(optimizer_args) if optimizer_args else None,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
            "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
        },
        "dataset_arguments": {
            "cache_latents": True,
            "debug_dataset": False,
            "vae_batch_size": vae_batch_size,
        },
        "training_arguments": {
            "output_dir": OUTPUT_DIR,
            "output_name": PROJECT_NAME,
            "save_precision": save_precision,
            "save_every_n_epochs": save_n_epochs_type_value if save_n_epochs_type == "save_every_n_epochs" else None,
            "save_n_epoch_ratio": save_n_epochs_type_value if save_n_epochs_type == "save_n_epoch_ratio" else None,
            "save_last_n_epochs": None,
            "save_state": None,
            "save_last_n_epochs_state": None,
            "resume": None,
            "train_batch_size": train_batch_size,
            "max_token_length": 225,
            "mem_eff_attn": False,
            "xformers": True,
            "max_train_epochs": num_epochs,
            "max_data_loader_n_workers": 8,
            "persistent_data_loader_workers": True,
            "seed": seed if seed > 0 else None,
            "gradient_checkpointing": gradient_checkpointing,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "mixed_precision": mixed_precision,
            "clip_skip": clip_skip if not v2 else None,
            "logging_dir": LOGGING_DIR,
            "log_prefix": PROJECT_NAME,
            "noise_offset": noise_offset if noise_offset > 0 else None,
            "lowram": lowram,
        },
        "sample_prompt_arguments": {
            "sample_every_n_steps": None,
            "sample_every_n_epochs": 1 if enable_sample_prompt else 999999,
            "sample_sampler": sampler,
        },
        "dreambooth_arguments": {
            "prior_loss_weight": 1.0,
        },
        "saving_arguments": {
            "save_model_as": save_model_as
        },
    }

    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None

        config_str = toml.dumps(config)

        def write_file(filename, contents):
            with open(filename, "w") as f:
                f.write(contents)

        write_file(CONFIG_PATH, config_str)
        write_file(SAMPLE_PROMPT_PATH, sample_str)

        # print(config_str)

        # @title ## 5.5. Start Training

    accelerate_conf = {
        "config_file": ACCELERATE_CONFIG_PATH,
        "num_cpu_threads_per_process": 1,
    }

    train_conf = {
        "sample_prompts": SAMPLE_PROMPT_PATH,
        "dataset_config": DATASET_CONFIG_PATH,
        "config_file": CONFIG_PATH
    }

    accelerate_args = train(accelerate_conf)
    train_args = train(train_conf)
    final_args = f"accelerate launch {accelerate_args} train_network.py {train_args}"

    os.system('conda activate art_env')
    os.system("conda train_network.py train_args")
    # os.chdir(repo_dir)
    # !{final_args}


if __name__ == "__main__":
    main()
