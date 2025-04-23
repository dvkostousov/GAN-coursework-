import matplotlib.pyplot as plt

from parameters import GAN_num_epochs

num_epochs = GAN_num_epochs

f = open('losses.txt', 'r')
a = list(map(float, f.readline().split()))
b = list(map(float, f.readline().split()))
plt.plot(range(num_epochs), a, label='discriminator')
plt.plot(range(num_epochs), b, label='generator')
plt.title('Losses of generator and discriminator in the training process')
plt.ylabel('loss')
plt.xlabel('epoch number')
plt.legend()
plt.show()
