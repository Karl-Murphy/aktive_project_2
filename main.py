import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import random

# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    initial_training_size: int = 1000
    pool_size: int = 10000
    batch_size: int = 50
    learning_rate: float = 1e-3
    pretrain_epochs: int = 10
    active_rounds: int = 50
    acquisition_size: int = 10
    num_models: int = 10
    num_classes: int = 10
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer = optim.Adam
    epoch: int = 3
    test_dataset_size = 1000



CFG = Config()

CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


# ============================================================
# Model
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(number_of_models: int) -> list[nn.Module]:
    models = []
    for i in range(number_of_models):
        set_seed(i)
        models.append(SimpleCNN(num_classes=CFG.num_classes).to(CFG.device))
    return models


# ============================================================
# Data
# ============================================================

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    return train_transform, test_transform


def build_datasets():
    train_transform, test_transform = get_transforms()

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    remaining = len(train_dataset) - CFG.initial_training_size
    train_set, rest = random_split(
        train_dataset,
        [CFG.initial_training_size, remaining],
        generator=torch.Generator().manual_seed(CFG.seed),
    )

    remaining_after_pool = remaining - CFG.pool_size
    pool_set, _ = random_split(
        rest,
        [CFG.pool_size, remaining_after_pool],
        generator=torch.Generator().manual_seed(CFG.seed),
    )


    remaining = len(test_dataset) - CFG.test_dataset_size
    test_set, _ = random_split(
        test_dataset,
        [CFG.test_dataset_size, remaining],
        generator=torch.Generator().manual_seed(CFG.seed),
    )

    return train_set, pool_set, test_set


def make_loader(dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
    )



def bootstrap_subset(dataset: Subset):
    B = []
    for i in range(len(dataset.indices)):
        B.append(random.choice(dataset.indices))
    return Subset(dataset.dataset,B)


def pool_to_train(chosen, train_set: Subset, pool_set: Subset):
    for i in sorted(chosen.tolist(), reverse=True):
        train_set.indices.append(pool_set.indices.pop(i))
    return train_set, pool_set


# ============================================================
# Train / Eval
# ============================================================

def train_one_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for i in range(epoch):
        for images, labels in loader:
            images = images.to(CFG.device)
            labels = labels.to(CFG.device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, 100.0 * total_correct / total_examples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(CFG.device)
        labels = labels.to(CFG.device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, 100.0 * total_correct / total_examples


# ============================================================
# Committee inference
# ============================================================

@torch.no_grad()
def predict_probabilities(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    model.eval()
    all_probs = []

    for images, _ in loader:
        images = images.to(CFG.device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)  # [N, C]


@torch.no_grad()
def committee_predictions(models: List[nn.Module], pool_loader: DataLoader) -> torch.Tensor:
    """
    Returns predicted class indices from each model.
    Shape: [M, N]
    """
    preds = []
    for model in models:
        probs = predict_probabilities(model, pool_loader)
        preds.append(probs.argmax(dim=1))
    return torch.stack(preds, dim=0)


def vote_entropy(predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    predictions: [M, N] class indices
    returns: [N] vote entropy
    """
    num_models, num_items = predictions.shape
    scores = torch.zeros(num_items, dtype=torch.float32)

    for i in range(num_items):
        votes = torch.bincount(predictions[:, i], minlength=num_classes).float()
        probs = votes / num_models
        nonzero = probs > 0
        scores[i] = -(probs[nonzero] * torch.log2(probs[nonzero])).sum()

    return scores


def majority_vote(predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    predictions: [M, N]
    returns: [N]
    """
    num_items = predictions.shape[1]
    final_preds = []

    for i in range(num_items):
        votes = torch.bincount(predictions[:, i], minlength=num_classes)
        final_preds.append(votes.argmax())

    return torch.stack(final_preds)


# ============================================================
# Active learning
# ============================================================

def select_top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.numel())
    return torch.topk(scores, k=k).indices


def active_learning_round(models: List[nn.Module], pool_set: Subset) -> Tuple[torch.Tensor, torch.Tensor]:
    pool_loader = make_loader(pool_set, shuffle=False)

    committee_preds = committee_predictions(models, pool_loader)
    voted_classes = majority_vote(committee_preds, CFG.num_classes)

    acquisition_scores = vote_entropy(committee_preds,CFG.num_classes)

    chosen = select_top_k(acquisition_scores, CFG.acquisition_size)

    return chosen, voted_classes

# ============================================================
# Main
# ============================================================

def main():
    train_set, pool_set, test_dataset = build_datasets()

    test_loader = make_loader(test_dataset, shuffle=False)

    models = build_model(CFG.num_models)

    for model in models:
        bootstrapped_subset = bootstrap_subset(train_set)
        data = make_loader(bootstrapped_subset, shuffle=True)
        train_one_model(model,data, CFG.criterion, CFG.optimizer(params=model.parameters(),lr=CFG.learning_rate), CFG.pretrain_epochs)

    for _ in range(CFG.active_rounds):

        chosen, voted_classes = active_learning_round(models,pool_set)

        train_set, pool_set = pool_to_train(chosen, train_set, pool_set)


        for model in models:
            bootstrapped_subset = bootstrap_subset(train_set)
            data = make_loader(bootstrapped_subset, shuffle=True)
            train_one_model(model,data, CFG.criterion, CFG.optimizer(params=model.parameters(),lr=CFG.learning_rate), CFG.epoch)


        for i, model in enumerate(models):
            loss_pr_example, correct_precentage = evaluate(model,test_loader,CFG.criterion)
            print(f"loss of model {i} is {loss_pr_example} pr example and accuracy is {correct_precentage}")

if __name__ == "__main__":
    main()