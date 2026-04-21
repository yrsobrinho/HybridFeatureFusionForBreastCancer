import numpy as np
from PIL import Image

# Carrega o arquivo baixado
data = np.load('breastmnist.npz')
images = data['train_images']

# Salva a primeira imagem
img = Image.fromarray(images[0])
img.save('breast_sample.png')