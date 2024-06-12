import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


from model import KANMixer
from config import get_config, get_weights_file_path


# Get accuracy on training & test to see how good our model is
def get_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            _, predictions = logits.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return num_correct / num_samples


config = get_config()

Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(config["in_channels"])],
            [0.5 for _ in range(config["in_channels"])],
        ),
    ]
)

# MNIST Dataset
trainset = torchvision.datasets.MNIST(
    root="./data/mnist", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root="./data/mnist", train=False, download=True, transform=transform
)

# CIFAR10 Dataset
# trainset = torchvision.datasets.CIFAR10(
#     root="./data/cifar10", train=True, download=True, transform=transform
# )
# testset = torchvision.datasets.CIFAR10(
#     root="./data/cifar10", train=False, download=True, transform=transform
# )

train_loader = DataLoader(
    trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2
)
test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


model = KANMixer(
    in_channels=config["in_channels"],
    image_size=config["image_size"],
    patch_size=2,
    num_classes=10,
    embedding_dim=config["channel_dim"],
    depth=config["depth"],
    token_intermediate_dim=config["token_dim"],
    channel_intermediate_dim=config["channel_dim"],
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

num_epochs = config["num_epochs"]

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_index, (images, targets) in loop:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # progress bar information updating
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

    print(f"Accuracy on training set: {get_accuracy(train_loader, model)*100:.2f}")
    print(f"Accuracy on test set: {get_accuracy(test_loader, model)*100:.2f}")

    model_filename = get_weights_file_path(config, f"{epoch+1}")

    torch.save(
        {
            # "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "global_Step": global_step,
        },
        model_filename,
    )
