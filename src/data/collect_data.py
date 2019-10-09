import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import alexnet, vgg11, mobilenet, mnasnet0_5
from ai.ai import ArithmeticIntensity
from thop import profile
import timeit

models = [mobilenet.mobilenet_v2(), alexnet(), vgg11(), mnasnet0_5()]

print(alexnet().__str__())

metrics = []

for model in models:
    ai_profiler = ArithmeticIntensity(model=model, input_dims=(1, 3, 224, 224))
    ai, macs = ai_profiler.get_metrics()
    tensor = torch.randn(5, 3, 224, 224)
    _, params = profile(model, inputs=(tensor, ))
    def forward_pass():
        model(tensor)
    fpt = timeit.timeit(stmt=forward_pass, number=5)
    metrics.append([params, macs, ai, fpt])

params = [i[0] for i in metrics]
flops = [i[1] for i in metrics]
ais = [i[2] for i in metrics]
latencies = [i[3] for i in metrics]

plt.subplot(131)
x, y = zip(*sorted(zip(latencies, params)))
plt.plot(x, y)
plt.subplot(132)
x, y = zip(*sorted(zip(latencies, flops)))
plt.plot(x, y)
plt.subplot(133)
x, y = zip(*sorted(zip(latencies, ais)))
plt.plot(x, y)
plt.suptitle("Comparision with latencies")

plt.show()