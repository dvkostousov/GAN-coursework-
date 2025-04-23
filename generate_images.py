import torch
from torchvision.utils import save_image

from discriminator import Discriminator
from generator import Generator
from parameters import GAN_z_dim, GAN_batch_size, num_of_images

batch_size = GAN_batch_size
z_dim = GAN_z_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка моделей
generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.load_state_dict(torch.load('generators/my_generator_99.pth'))
discriminator.load_state_dict(torch.load('discriminators/my_discriminator_99.pth'))

f = open('synth_women.txt', 'w')
for image_num in range(num_of_images):
    z = torch.randn(batch_size, z_dim, device=device)

    with torch.no_grad():
        # Генерация изображений
        fake_images = generator(z)  # shape: [64, 3, 128, 128]
        # Вычисление оценок дискриминатора для каждого изображения
        scores = discriminator(fake_images)  # shape: [64, 1]
        scores = scores.view(-1)
        # Находим индекс изображения с максимальной оценкой
        best_index = torch.argmax(scores)
        best_image = fake_images[best_index]

    # Сохраняем выбранное изображение
    save_image(best_image, f'celeba/img_align_celeba/image_{image_num}.png', normalize=True)
    print(f"image_{image_num}.png", file=f)
    print(f"Сохранено изображение image_{image_num}.png с оценкой дискриминатора:", scores[best_index].item())
f.close()
