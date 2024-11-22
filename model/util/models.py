import torch
import torch.nn as nn
import math

class Flatten(torch.nn.Module):
    def __init__(self, dims):
        super(Flatten, self).__init__()
        self.dims = dims

    def forward(self, x):
        dim = 1
        for d in self.dims:
            dim *= x.shape[d]
        return x.reshape(-1, dim)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"
class ConvEnc(torch.nn.Module):
    def __init__(self, channel_list, hidden_dim, out_dim):
        super(ConvEnc, self).__init__()
        convs = []
        for i in range(len(channel_list)-1):
            convs.append(Conv3x3Block(channel_list[i], channel_list[i+1], sampling="down"))
        self.conv = torch.nn.Sequential(
            torch.nn.Sequential(*convs),
            Flatten([1, 2, 3]),
            torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.conv(x)
class ConvDec(torch.nn.Module):
    def __init__(self, channel_list, in_dim, hidden_dim):
        super(ConvDec, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU())
        self.conv = []
        for i in range(len(channel_list)-1):
            self.conv.append(Conv3x3Block(channel_list[i], channel_list[i+1], sampling="up"))

        self.conv.append(torch.nn.Conv2d(channel_list[-1], 16, kernel_size=3, stride=1, padding=1))
        self.conv.append(torch.nn.ReLU())
        self.conv.append(torch.nn.Conv2d(16, 12, kernel_size=3, stride=1, padding=1))
        self.conv.append(torch.nn.ReLU())
        self.conv.append(torch.nn.Conv2d(12, 2, kernel_size=3, stride=1, padding=1))
        # self.conv.append(torch.nn.Tanh())
        self.conv = torch.nn.Sequential(*self.conv)
        self.first_filter = self.conv[0].conv[0].weight.shape[1]

    def forward(self, x):
        h = self.fc(x)
        width = int(math.sqrt(h.shape.numel() // (x.shape[0] * self.first_filter)))
        h = h.reshape(x.shape[0], self.first_filter, width, width)
        h=self.conv(h)
        
        return h
class Conv3x3Block(nn.Module):
    def __init__(self,in_channel,out_chanel,sampling):
        super(Conv3x3Block,self).__init__()
        self.conv=[nn.Conv2d(in_channel,out_chanel,kernel_size=3,stride=1,padding=1)]
        self.conv.append(nn.ReLU())
        if sampling=='down':
            self.conv.append(nn.MaxPool2d(kernel_size=2))
        elif sampling=='up':
            self.conv.append(nn.Upsample(scale_factor=2,mode='nearest'))
        self.conv=nn.Sequential(*self.conv)
    def forward(self,x):
        return self.conv(x)
    
class MLP(torch.nn.Module):
    def __init__(self, layer_info, activation="relu", init="he"):
        super(MLP, self).__init__()

        if activation == "relu":
            func = torch.nn.ReLU()
        elif activation == "tanh":
            func = torch.nn.Tanh()
        
        if init == "xavier":
            gain = 1.0
        else:
            gain = torch.nn.init.calculate_gain(activation)

        layers = []
        in_dim = layer_info[0]
        for i, unit in enumerate(layer_info[1:-1]):
            layers.append(Linear(in_features=in_dim, out_features=unit, gain=gain, init=init))
            layers.append(func)
            in_dim = unit
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], gain=1.0, init=init))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        out=self.layers(x)
        
        return out


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, gain=1.0, init="he"):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))

        if init == "he":
            d = (self.weight.size(1))
        elif init == "xavier":
            d = (self.weight.size(0) + self.weight.size(1)) / 2
        stdv = gain * math.sqrt(1 / d)
        self.weight.data.normal_(0., stdv)
        self.bias.data.zero_()

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)
