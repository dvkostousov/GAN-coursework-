import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from discriminator import Discriminator
from generator import Generator

from parameters import GAN_batch_size, GAN_lr, beta1, beta_2, GAN_num_epochs, GAN_z_dim, start_epoch_to_save


class CustomDataset(Dataset):
    def __init__(self, image_dir, images_file, transform=None):

        self.image_dir = image_dir
        self.images_file = images_file
        self.transform = transform
        self.image_paths = []

        with open(images_file, 'r') as f:
            for line in f:
                print(line[:-1])
                self.image_paths.append(os.path.join(image_dir, line[:-1]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


batch_size = GAN_batch_size
lr = GAN_lr
num_epochs = GAN_num_epochs
z_dim = GAN_z_dim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Уменьшаем изображения до 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация
])

# Путь к данным
dir_with_images = "celeba\img_align_celeba"
file_with_images_name = 'women.txt'
dataset = CustomDataset(dir_with_images, file_with_images_name, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Инициализация модели, оптимизаторов и потерь
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta_2))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta_2))

# Обучение GAN
fixed_noise = torch.randn(batch_size, z_dim).to(device)
avg_d_losses = []
avg_g_losses = []
num_steps = 8

for epoch in range(num_epochs):
    sum_g_loss = 0
    sum_d_loss = 0
    for i, real_images in enumerate(dataloader):
        if i > num_steps:
            break
        # Подготовка данных
        real_images = real_images.to(device)

        # Генерация шума
        z = torch.randn(real_images.size(0), z_dim).to(device)

        # Обучение дискриминатора
        optimizer_d.zero_grad()

        # Реальные изображения
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        real_output = discriminator(real_images)
        d_loss_real = criterion(real_output, real_labels)

        # Ложные изображения
        fake_images = generator(z)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
        fake_output = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)

        # Общая потеря дискриминатора
        d_loss = d_loss_real + d_loss_fake
        sum_d_loss += d_loss.item()
        d_loss.backward()
        optimizer_d.step()

        # Обучение генератора
        optimizer_g.zero_grad()

        # Потеря генератора
        g_loss = criterion(discriminator(fake_images), real_labels)
        sum_g_loss += g_loss.item()
        g_loss.backward()
        optimizer_g.step()

        # Печать потерь
        print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{num_steps}], "
              f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    avg_g_losses.append(sum_g_loss / num_steps)
    avg_d_losses.append(sum_d_loss / num_steps)
    # Генерация изображений после каждой эпохи
    with torch.no_grad():
        fake_images = generator(fixed_noise).to(device)
        save_image(fake_images, f"generated_images/epoch_{epoch}.png", nrow=8, normalize=True)

    # Сохранение обученных моделей генератора и дискриминатора
    if epoch >= start_epoch_to_save:
        torch.save(generator.state_dict(), f'generators/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'discriminators/discriminator_{epoch}.pth')

# Запись потерь во время обучения
log_loss_file = open('losses.txt', 'w')
print(' '.join([str(i) for i in avg_d_losses]), file=log_loss_file)
print(' '.join([str(i) for i in avg_g_losses]), file=log_loss_file)
