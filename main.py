import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import sympy as sp

Initial_training = 20
train_pool = 20
BATCH_SIZE = 5
LEARNING_RATE = 1e-3
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs_active = 3

# ----------------------------
# Data transforms
# ----------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])



# ----------------------------
# CIFAR-10 datasets and loaders
# ----------------------------
train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

rest_int = len(train_dataset)  - Initial_training

Initial_train, rest  = random_split(train_dataset, [Initial_training, rest_int])

rest_int2 = rest_int - train_pool

pool, rest  = random_split(rest, [train_pool, rest_int2])


train_loader = torch.utils.data.DataLoader(
    Initial_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

pool_loader = torch.utils.data.DataLoader(
    pool,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)



classes = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)    # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

# ----------------------------
# Save model
# ----------------------------
torch.save(model.state_dict(), "cifar10_cnn.pth")
print("Model saved to cifar10_cnn.pth")

models = [model]

#pre train

for model in models:
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )


#active loop

for i in range(epochs_active):
    model_pred = [[] for i in range(len(models))]
    for i, model in enumerate(models):
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            for image, label in pool:
                image = image.unsqueeze(0).to(DEVICE)  # (1, 3, 32, 32)
                out = model(image)                     # (1, 10) logits
                probs = torch.softmax(out, dim=1)      # softmax over class dimension

                pred_idx = probs.argmax(dim=1).item()
                pred_label = classes[pred_idx]
                model_pred[i].append(pred_label)
    QBC_predictions = []
    for i in range(len(pool)):
        counts = []
        for C in classes:
            count = 0
            for model in model_pred:
                if model[i] == C:
                    count += 1
            counts.append((count,C))
        QBC_predictions.append(max(counts))

QBC_predictions

model_pred = [[] for model in models]
for i, model in enumerate(models):
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for image, label in pool:
            image = image.unsqueeze(0).to(DEVICE)  # (1, 3, 32, 32)
            out = model(image)                     # (1, 10) logits
            probs = torch.softmax(out, dim=1)      # softmax over class dimension

            pred_idx = probs.argmax(dim=1).item()
            true_label = classes[label]
            pred_label = classes[pred_idx]
            pred_conf = probs[0, pred_idx].item()

            print(f"true={true_label:>6} | pred={pred_label:>6} | conf={pred_conf:.2%}")
            print(probs.squeeze(0).cpu())
            model_pred[i].append(probs)

#pre train

for model in models:
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )


#active loop

for i in range(epochs_active):
    model_pred = [[] for i in range(len(models))]
    for i, model in enumerate(models):
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            for image, label in pool:
                image = image.unsqueeze(0).to(DEVICE)  # (1, 3, 32, 32)
                out = model(image)                     # (1, 10) logits
                probs = torch.softmax(out, dim=1)      # softmax over class dimension

                pred_idx = probs.argmax(dim=1).item()
                pred_label = classes[pred_idx]
                model_pred[i].append(pred_label)
    QBC_predictions = []
    QBC_vote_entropy = []
    for i in range(len(pool)):
        counts = []
        vote_entropy = []
        for C in classes:
            count = 0
            for model in model_pred:
                if model[i] == C:
                    count += 1
            counts.append(count)
            vote_entropy.append((count/len(models) * sp.log(count/len(models),2),C))
        
        QBC_predictions.append(max(counts))
        QBC_vote_entropy.append(sum(vote_entropy))
