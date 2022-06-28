'''
Imports
'''
import matplotlib.pyplot as plt
import numpy as np


'''
Function to display an image:
'''
def display_image(image_tensor):
  cmap = input(f'Digite 1 si desea mostrar imagen en RGB o 2 para escala de grises:  ')
  cmap = ('gray' if cmap == '2' else None)
  tensor_image = image_tensor.view(image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[0])
  plt.imshow(tensor_image, cmap=cmap)
  plt.show()

