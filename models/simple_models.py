import freezable_modules.modules as mod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""l = mod.FreezableLinear(10,10,freeze_matrix=torch.as_tensor([True, False, False, False, False, False, False, False, False, False]))
with torch.no_grad():
    print(l.neuron_tensor[0])
    print("changing 1st tensor")
    l.neurons[0] += 1
    print(l.neurons[0])
    print(l.neuron_tensor[0])"""

class LeNet(mod.FreezableModule):
    def __init__(self, n_output):
        super(LeNet, self).__init__()
        self.c1 = mod.FreezableConv2d(1,6,5,5)
        self.c2 = mod.FreezableConv2d(6,16,5,5)
        self.classifier = mod.SequentialF(mod.FreezableLinear(256,120),
                                        nn.Sigmoid(),
                                        mod.FreezableLinear(120,84),
                                        nn.Sigmoid(),
                                        mod.FreezableLinear(84,n_output),
                                        nn.LogSoftmax(dim=1))
    
    def forward(self,x):
        x = self.c1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2, stride=2)
        x = self.c2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2, stride=2)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class AlexNet(mod.FreezableModule):
    def __init__(self, n_output):
        super(AlexNet, self).__init__()
        self.features = mod.SequentialF(
            mod.FreezableConv2d(3, 96, 7, 7, True, 0, 2),
            nn.ReLU(),
            mod.FreezableConv2d(96, 256, 5, 5, True, 1),
            nn.ReLU(),
            mod.FreezableConv2d(256, 384, 3, 3, True, 1),
            nn.ReLU(),
            mod.FreezableConv2d(384, 384, 3, 3, True, 1),
            nn.ReLU(),
            mod.FreezableConv2d(384, 256, 3, 3, True, 1),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        self.classifier = mod.SequentialF(
            mod.FreezableLinear(6400, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            mod.FreezableLinear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            mod.FreezableLinear(512, n_output),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x