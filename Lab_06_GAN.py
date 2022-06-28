'''
    Universidad de Costa Rica
    Machine Learning
    Laboratorio 06, GAN (generative adversarial network)
    Daniel Ricardo Ramírez Umaña, B45675
'''
from telnetlib import STATUS
import torch
import torchvision
import utilities


'''
Data loading
'''
train = torchvision.datasets.MNIST(".", download=True)
x = train.data.float()
y = train.targets


#Verificar si estamos usando GPU. 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("El modelo estará corriendo en ", device)
   

'''
We need to add a dimension
  from: N, H, W
  to:   N, C, H, W
where:
  N:    sample number
  C:    number of channels (RGB or graysclae)
  H:    height
  W:    width
'''
x = torch.unsqueeze(x, 1)

utilities.display_image(x[1])

print("Programa ha terminado")
print(type(x))
