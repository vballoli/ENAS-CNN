import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import alexnet, vgg11, mobilenet, mnasnet0_5, resnet18, resnet34, mobilenet_v2, densenet121, shufflenet_v2_x0_5, squeezenet1_0, vgg19
from ai.ai import ArithmeticIntensity
from thop import profile
import timeit

models = [alexnet(), resnet18(), resnet34(), mobilenet_v2(), densenet121(), shufflenet_v2_x0_5(), squeezenet1_0(), vgg19()]
model_names = [alexnet.__name__, resnet18.__name__, resnet34.__name__, mobilenet_v2.__name__, densenet121.__name__, shufflenet_v2_x0_5.__name__, squeezenet1_0.__name__, vgg19.__name__]
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
x, y, z = zip(*sorted(zip(params, latencies, model_names)))
print("P: ", z)
plt.plot(x, y)
plt.xlabel("Parameters")
plt.ylabel("Latency")
plt.subplot(132)
x, y, z = zip(*sorted(zip(flops, latencies, model_names)))
print("M: ", z)
plt.plot(x, y)
plt.ylabel("Latency")
plt.xlabel("FLOPS")
plt.subplot(133)
x, y, z = zip(*sorted(zip(ais, latencies, model_names)))
print("AI: ", z)
plt.plot(x, y)
plt.ylabel("Latency")
plt.xlabel("Arithmetic Intensity")
plt.suptitle("Comparision with latencies")

plt.show()