from pathlib import Path


def get_config():
    return {
        "in_channels": 1,  # change to 3 if you use CIFAR10 dataset (1 for MNIST)
        "image_size": 28,  # change to 32 if you use CIFAR10 dataset (28 for MNIST)
        "num_classes": 10,
        "learning_rate": 4e-3,
        "batch_size": 128,
        "num_epochs": 1,  # 25
        "channel_dim": 128,
        "token_dim": 64,
        "depth": 4,
        "model_folder": "weights",
        "model_basename": "mixer-model_",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
