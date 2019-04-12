import torch
from math import floor


class LosslessPooling(torch.nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=0,
                 stride=1, shuffle=True):
        super(LosslessPooling, self).__init__()
        self.kernel_size = kernel_size \
            if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.shuffle = shuffle
        self.unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)

    def get_shape(self, h_w):
        h, w = list(map(lambda i: floor(((h_w[i] + (2*self.padding) - \
                                (self.dilation*(self.kernel_size[i]-1))-1)\
                                /self.stride) + 1),
                        range(2)))
        return h, w

    def forward(self, x):
        x_unf = self.unfold(x)
        x_out = x_unf.view(
            x.shape[0],
            x.shape[1] * self.kernel_size[0] * self.kernel_size[1],
            *self.get_shape(x.shape[2:])
        )
        if self.shuffle:
            return x_out[:, torch.randperm(x_out.shape[1])]
        return x_out

