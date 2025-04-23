import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

from parameters import CNN_num_epochs, CNN_lr, CNN_batch_size


class CustomDataset(Dataset):
    def __init__(self, image_dir, women_txts, men_txt, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.women_list = []
        for txt in women_txts:
            with open(txt, 'r', encoding='utf-8') as f:
                self.women_list.extend([line.strip() for line in f if line.strip()])

        with open(men_txt, 'r', encoding='utf-8') as f:
            self.men_list = [line.strip() for line in f if line.strip()]

        self.data = [(os.path.join(image_dir, fname), 0) for fname in self.women_list] + \
                    [(os.path.join(image_dir, fname), 1) for fname in self.men_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Ошибка при открытии {img_path}: {e}")
            return torch.zeros((3, 128, 128)), label
        return image, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, dataloader, num_epochs=CNN_num_epochs, learning_rate=CNN_lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader.dataset):.4f}")
    return model


def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.numpy()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)

    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds))
    print("Recall:", recall_score(all_labels, all_preds))
    print("F1-score:", f1_score(all_labels, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=["Women", "Men"]))


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

image_directory = 'celeba/img_align_celeba/'
real_women_txt, synth_women_txt, men_1_txt, men_2_txt = 'real_women_2.txt', 'synth_women.txt', 'men_1.txt', \
    'men_2.txt'

dataset_stage1 = CustomDataset(image_directory, [real_women_txt], men_1_txt, transform)
loader_stage1 = DataLoader(dataset_stage1, batch_size=CNN_batch_size, shuffle=True, num_workers=4)

dataset_stage2 = CustomDataset(image_directory, [real_women_txt], men_2_txt, transform)
loader_stage2 = DataLoader(dataset_stage2, batch_size=CNN_batch_size, shuffle=True, num_workers=4)

dataset_stage3 = CustomDataset(image_directory, [real_women_txt, synth_women_txt], men_2_txt, transform)
loader_stage3 = DataLoader(dataset_stage3, batch_size=CNN_batch_size, shuffle=True, num_workers=4)

test_women_txt, test_men_txt = 'test_women.txt', 'test_men.txt'
test_dataset = CustomDataset(image_directory, [test_women_txt], test_men_txt, transform)
test_loader = DataLoader(test_dataset, batch_size=CNN_batch_size    , shuffle=False, num_workers=4)

print("Stage 1: Training on real data")
model_stage1 = train_model(SimpleCNN(), loader_stage1)
print("\nEvaluation of model trained on real data:")
evaluate_model(model_stage1, test_loader)

print("Stage 2: Training on real data with class disbalance")
model_stage2 = train_model(SimpleCNN(), loader_stage2)
print("\nEvaluation of model trained on real data:")
evaluate_model(model_stage2, test_loader)

print("\nStage 3: Training on combined data")
model_stage3 = train_model(SimpleCNN(), loader_stage3)
print("\nEvaluation of model trained on combined data:")
evaluate_model(model_stage3, test_loader)
