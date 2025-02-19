import torch
import torch.nn as nn
import torch.nn.functional as F



def _get_act(s_act):
    if s_act == "relu":
        return nn.ReLU(inplace=True)
    elif s_act == "sigmoid":
        return nn.Sigmoid()
    elif s_act == "softplus":
        return nn.Softplus()
    elif s_act == "linear":
        return None
    elif s_act == "tanh":
        return nn.Tanh()
    elif s_act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == "softmax":
        return nn.Softmax(dim=1)
    elif s_act == "selu":
        return nn.SELU()
    elif s_act == "elu":
        return nn.ELU()
    elif s_act == "gelu":
        return nn.GELU()
    elif s_act == "prelu":
        return nn.PReLU()
    elif s_act == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unexpected activation: {s_act}")


def conv1d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def conv2d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class CNN1D(nn.Module) :
    def __init__(self, act, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(CNN1D,self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.kern = kernel_size
        self.pad = padding
        self.act = _get_act(act)
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.conv_list = []
        if self.blocks != 0:
            for block in range(self.blocks):
                self.conv_list.append(conv1d(filters, filters, kernel_size=self.kern, padding=self.pad))
                self.conv_list.append(self.act)
        self.conv_list=nn.Sequential(*self.conv_list)
        self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*self.d_out, self.d_out, bias=True)
    def forward(self, x):
        out = self.act(self.conv1(x))
        if self.blocks != 0:
            out = self.conv_list(out)
        out = self.convH(out)
        out = out.flatten(start_dim=1)     
        out = self.fcH(out)       
        out = out.view(out.shape[0], 1, self.d_out)        
        return out
    


class CNN2D(nn.Module) : # Linear
    def __init__(self, act, resol_in, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(CNN2D, self).__init__()
        self.resol_in = resol_in
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.act = _get_act(act)
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv2d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.conv_list = []
        if self.blocks != 0:
            for block in range(self.blocks):
                self.conv_list.append(conv2d(filters, filters, kernel_size=self.kern, padding=self.pad))
                self.conv_list.append(self.act)
        self.conv_list=nn.Sequential(*self.conv_list)
        self.convH = conv2d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*resol_in**2, self.d_out, bias=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        if self.blocks != 0:
            out = self.conv_list(out)
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fcH(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out