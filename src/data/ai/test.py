import torch
from torchvision.models import alexnet, vgg11, mobilenet, inception
from ai import ArithmeticIntensity
model = vgg11()
print(model)
ai = ArithmeticIntensity(model=model, input_dims=(1, 3, 224, 224))
ai.get_metrics()