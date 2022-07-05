"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

import numpy as np
from tensorboardX import SummaryWriter

import os
import time
import math
import argparse
import pprint

import librosa
import random

parser = argparse.ArgumentParser()
# action
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--visualize', action='store_true', help='Visualize manipulated attribures.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--seed', type=int, help='Random seed to use.')
# paths and reporting
parser.add_argument('--train_dir',  help='Location of training audios.')
parser.add_argument('--val_dir', help='Location of validation audios.')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--log_interval', type=int, default=2, help='How often to show loss statistics and save samples.')
parser.add_argument('--save_interval', type=int, default=50, help='How often to save during training.')
parser.add_argument('--eval_interval', type=int, default=1, help='Number of epochs to eval model and save model checkpoint.')
# data
parser.add_argument('--lazy', default=True, type=bool, help='Load audios on the fly?')
parser.add_argument('--sr', default=16000 ,type=int, help='Sample Rate')
parser.add_argument('--n_fft', default = 2048, type=int, help='Number of points in FFT for the STFT')
parser.add_argument('--win_length', default=2000, type=int, help='STFT Window application length')
parser.add_argument('--hop_length', default=500,type=int, help='STFT Window hop length')
# model parameters
parser.add_argument('--depth', type=int, default=2, help='Depth of the network (cf Glow figure 2).')
parser.add_argument('--n_levels', type=int, default=2, help='Number of levels of of the network (cf Glow figure 2).')
parser.add_argument('--z_std', type=float, help='Pass specific standard devition during generation/sampling.')
# training params
parser.add_argument('--batch_size', type=int, default=1, help='Training batch size.')
parser.add_argument('--batch_size_init', type=int, default=1, help='Batch size for the data dependent initialization.')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--n_epochs_warmup', type=int, default=2, help='Number of warmup epochs for linear learning rate annealing.')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--mini_data_size', type=int, default=None, help='Train only on this number of datapoints.')
parser.add_argument('--grad_norm_clip', default=50, type=float, help='Clip gradients during training.')
parser.add_argument('--checkpoint_grads', action='store_true', default=False, help='Whether to use gradient checkpointing in forward pass.')
parser.add_argument('--n_bits', default=5, type=int, help='Number of bits for input images.')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
# distributed training params
parser.add_argument('--distributed', action='store_true', default=False, help='Whether to use DistributedDataParallels on multiple machines and GPUs.')
parser.add_argument('--world_size', type=int, default=1, help='Number of nodes for distributed training.')
parser.add_argument('--local_rank', type=int, help='When provided, run model on this cuda device. When None, used by torch.distributed.launch utility to manage multi-GPU training.')
# visualize
parser.add_argument('--vis_img', type=str, help='Path to image file to manipulate attributes and visualize.')
parser.add_argument('--vis_attrs', nargs='+', type=int, help='Which attribute to manipulate.')
parser.add_argument('--vis_alphas', nargs='+', type=float, help='Step size on the manipulation direction.')


best_eval_logprob = float('-inf')


# --------------------
# Dataset
# --------------------
class AudioTupleDataset(Dataset):
    def __init__(self, path, sr, lazy, sample_size, seed, n_fft, win_length, hop_length):
        super().__init__()
        self._sr = sr
        self._lazy = lazy
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self._path_tuples = []
        data_path = os.path.dirname(path)

        # Get path of tuples
        with open(path) as f:
            for line in f:
                path_tuple = line.rstrip('\n').split('\t')
                path_tuple = tuple(os.path.join(data_path, p) for p in path_tuple)
                self._path_tuples.append(path_tuple)

        # Subset
        if sample_size:
            self._path_tuples = random.Random(seed).sample(
                self._path_tuples, sample_size)

        # Not on the fly loading
        if not self._lazy:
            self._data = [self._load(i) for i in range(len(self))]

    def __getitem__(self, index):
        if self._lazy:
            return self._load(index)
        else:
            return self._data[index]

    def _load(self, index):
        audios = (librosa.load(path, sr=self._sr)[0]
                  for path in self._path_tuples[index])
        audios = (self.audio_to_mel_spec(audio) for audio in audios)
        return tuple(audios)

    def audio_to_mel_spec(self, audio):
        if len(audio) == 0:
            audio = np.zeros(shape=[1], dtype=audio.dtype)
        return np.log1p(np.abs(librosa.stft(y=audio, n_fft = self.n_fft, win_length = self.win_length, hop_length = self.hop_length)))

    def __len__(self):
        return len(self._path_tuples)


# --------------------
# Dataloader
# --------------------
def fetch_dataloader(args, train=True, data_dependent_init=False):
    # Dataset of unprocessed audios
    dataset = AudioTupleDataset(path=(args.train_dir if train else args.val_dir), sr=args.sr, lazy=(True if (train and args.lazy) else False), 
    sample_size=args.mini_data_size, seed=args.seed, n_fft=args.n_fft, win_length=args.win_length, hop_length=args.hop_length)

    args.input_dims = (1,1025,1)

    # if data dependent init use init batch size
    batch_size = args.batch_size_init if data_dependent_init else args.batch_size  

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.device.type == 'cuda' else {}
    return DataLoader(dataset = dataset, collate_fn=collate_padded_tuples, shuffle= (True if train else False), batch_size=batch_size, **kwargs)

def collate_padded_tuples(batch):
    batch = tuple(zip(*batch))
    lengths = [[x.shape[1] for x in inputs] for inputs in batch]
    max_lengths = [max(x) for x in lengths]
    batch = [[np.pad(x, [(0, 0), (0, max(0, max_len - x.shape[1]))]) for x in inputs]
             for inputs, max_len in zip(batch, max_lengths)]
    return tuple((torch.as_tensor(np.array(x)), torch.as_tensor(np.array(l))) for x, l in zip(batch, lengths))

# --------------------
# Model component layers
# --------------------
class Actnorm(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.channels = channels
        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, input_dict, x_mask=None, reverse=False, **kwargs):
        x = input_dict['melspec']
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype=x.dtype)
            x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized += 1
        z = (self.bias + torch.exp(self.logs) * x) * x_mask
        logdet = torch.sum(self.logs) * x_len # [b]
        output_dict = {'melspec':z,'logdet':logdet, 'conditioning':input_dict['conditioning']}
        return output_dict

    def inverse(self, x):
        x = x['melspec']
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype=x.dtype)
            x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized += 1
        z = (x- self.bias) * torch.exp(-self.logs) * x_mask
        logdet = None
        return {'melspec':z,'logdet':logdet}

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m ** 2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
            logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)

class InvConvNear(nn.Module):
  def __init__(self, channels, n_split=2, no_jacobian=False, **kwargs):
    super().__init__()
    assert(n_split % 2 == 0)
    self.channels = channels
    self.n_split = n_split
    self.no_jacobian = no_jacobian
    
    w_init = torch.linalg.qr( torch.FloatTensor(self.n_split, self.n_split).normal_(), 'reduced')[0]
    if torch.det(w_init) < 0:
      w_init[:,0] = -1 * w_init[:,0]
    self.weight = nn.Parameter(w_init)

  def forward(self, input_dict, x_mask=None, reverse=False, **kwargs):
    x = input_dict['melspec']
    b, c, t = x.size()
    assert(c % self.n_split == 0)
    if x_mask is None:
      x_mask = 1
      x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
    else:
      x_len = torch.sum(x_mask, [1, 2])

    x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
    x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

    weight = self.weight
    if self.no_jacobian:
        logdet = 0
    else:
        logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len # [b]
    
    weight = weight.view(self.n_split, self.n_split, 1, 1)
    z = F.conv2d(x, weight)
    z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
    z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
    output_dict = {'melspec':z,'logdet':logdet, 'conditioning':input_dict['conditioning']}
    return output_dict

def reverse (self, x, x_mask=None, reverse=False, **kwargs):
    x = x['melspec']
    b, c, t = x.size()
    assert(c % self.n_split == 0)
    if x_mask is None:
      x_mask = 1
      x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
    else:
      x_len = torch.sum(x_mask, [1, 2])

    x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
    x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)
    
    if hasattr(self, "weight_inv"):
        weight = self.weight_inv
    else:
        weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
        logdet = None
    
    weight = weight.view(self.n_split, self.n_split, 1, 1)
    z = F.conv2d(x, weight)

    z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
    z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
    return {'melspec':z,'logdet':logdet}
        
class AffineCoupling(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1x1 Conv Layer
        self.conv1 = nn.Conv1d(1025, 1025, kernel_size=1)
        self.conv2 = nn.Conv1d(2050, 1025, kernel_size=1)

        # GTU (Gated Tanh Unit) Layer
        self.conv_2_s = nn.Conv1d(1025, 2050, kernel_size=1)
        self.conv_2_c = nn.Conv1d(1025, 2050, kernel_size=1)
        
        # 1x1 Conv Layer
        self.conv3 = nn.Conv1d(2050, 2050, kernel_size=1)        
        self.log_scale_factor = nn.Parameter(torch.zeros(2050,1,1))

        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()

    def forward(self, input_dict):
        inp = input_dict['melspec']
        cond = input_dict['conditioning']
        x_a, x_b = inp.chunk(2, 1)  # xa, xb = split(x) along channel dim 
        h_res = self.conv1(x_b) # h = NN(xb, style)
        cond = self.conv2(cond)
        h = torch.cat([h_res, cond], dim=2)
        h = torch.tanh(self.conv_2_c(h)) * torch.sigmoid(self.conv_2_s(h))
        h = self.conv3(h) * self.log_scale_factor.exp()

        # log(s), t = h
        m = h[:, :self.in_channels//2, :]
        logs = h[:, self.in_channels//2:, :]
        logs = torch.log(1e-6 + torch.sigmoid(logs + 2))

        z_b = x_b
        z_a = (m + torch.exp(logs) * x_a)
        logdet = torch.sum(logs, [1, 2])

        # z = concat(za,zb)
        z = torch.cat([z_a, z_b], dim=1)  # concat along channel dim

        output_dict = {'melspec':z,'logdet':logdet, 'conditioning':input_dict['conditioning']}
        print(z.shape)
        return output_dict

    def inverse(self, x):
        inp = x['melspec']
        cond = x['conditioning']
        z_a, z_b = inp.chunk(2, 1)  # split along channel dim

        h_res = self.conv1(z_b)

        h = torch.tanh(self.conv_2_c(h_res)) * torch.sigmoid(self.conv_2_s(cond))
        h = h + h_res
        h = self.conv3(h)  * self.log_scale_factor.exp()
        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        s = torch.sigmoid(s + 2.)

        x_a = (z_a - t) / s
        x_b = z_b
        x = torch.cat([x_a, x_b], dim=1)  # concat along channel dim

        logdet = - s.log().sum([1, 2, 3])
        output_dict = {'melspec':x,'logdet':logdet, 'conditioning':cond}
        return output_dict


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_dict, x_mask=None, n_sqz=2):
        x = input_dict['melspec']
        b, c, t = x.size()

        t = (t // n_sqz) * n_sqz
        x = x[:,:,:t]
        x_sqz = x.view(b, c, t//n_sqz, n_sqz)
        x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c*n_sqz, t//n_sqz)  

        if x_mask is not None:
            x_mask = x_mask[:,:,n_sqz-1::n_sqz]
        else:
            x_mask = torch.ones(b, 1, t//n_sqz).to(device=x.device, dtype=x.dtype)
        output_dict = {'melspec':x_sqz * x_mask,'logdet':0, 'conditioning':input_dict['conditioning']}
        return output_dict  

    def inverse(x, x_mask=None, n_sqz=2):
        x = x['melspec']
        b, c, t = x.size()

        x_unsqz = x.view(b, n_sqz, c//n_sqz, t)
        x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c//n_sqz, t*n_sqz)

        if x_mask is not None:
            x_mask = x_mask.unsqueeze(-1).repeat(1,1,1,n_sqz).view(b, 1, t*n_sqz)
        else:
            x_mask = torch.ones(b, 1, t*n_sqz).to(device=x.device, dtype=x.dtype)
        return x_unsqz * x_mask

class Split(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.gaussianize = Gaussianize(n_channels//2)

    def forward(self, x):
        x = x['melspec']
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim
        z2, logdet = self.gaussianize(x1, x2)
        return x1, z2, logdet

    def inverse(self, x1, z2):
        x2, logdet = self.gaussianize.inverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x, logdet


class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Conv2d(n_channels, 2*n_channels, kernel_size=3, padding=1)  # computes the parameters of Gaussian
        self.log_scale_factor = nn.Parameter(torch.zeros(2*n_channels,1,1))       # learned scale (cf RealNVP sec 4.1 / Glow official code
        # initialize to identity
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2):
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]          # split along channel dims
        z2 = (x2 - m) * torch.exp(-logs)                # center and scale; log prob is computed at the model forward
        logdet = - logs.sum([1,2,3])
        return z2, logdet

    def inverse(self, x1, z2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1,2,3])
        return x2, logdet

# --------------------
# Container layers
# --------------------

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def __init__(self, *args, **kwargs):
        self.checkpoint_grads = kwargs.pop('checkpoint_grads', None)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        sum_logdets = 0.
        for module in self:
            x = module(x) if not self.checkpoint_grads else checkpoint(module, x)
            sum_logdets = sum_logdets + x['logdet']
        return x, sum_logdets

    def inverse(self, z):
        sum_logdets = 0.
        for module in reversed(self):
            z, logdet = module.inverse(z)
            sum_logdets = sum_logdets + logdet
        return z, sum_logdets


class FlowStep(FlowSequential):
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """
    def __init__(self, n_channels):
        super().__init__(Actnorm(n_channels),
                         InvConvNear(n_channels),
                         AffineCoupling())


class FlowLevel(nn.Module):
    """ One depth level of Glow flow (Squeeze -> FlowStep x K -> Split); cf Glow figure 2b """
    def __init__(self, n_channels, depth, checkpoint_grads=False):
        super().__init__()
        self.squeeze = Squeeze()
        self.flowsteps = FlowSequential(*[FlowStep(2*n_channels) for _ in range(depth)], checkpoint_grads=checkpoint_grads)
        self.split = Split(2*n_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x, logdet_flowsteps = self.flowsteps(x)
        x1, z2, logdet_split = self.split(x)
        logdet = logdet_flowsteps + logdet_split
        return x1, z2, logdet

    def inverse(self, x1, z2):
        x, logdet_split = self.split.inverse(x1, z2)
        x, logdet_flowsteps = self.flowsteps.inverse(x)
        x = self.squeeze.inverse(x)
        logdet = logdet_flowsteps + logdet_split
        return x, logdet


# --------------------
# Model
# --------------------
class GlowTT(nn.Module):
    """ Glow multi-scale architecture with depth of flow K and number of levels L; cf Glow figure 2; section 3"""
    def __init__(self, depth, n_levels, input_dims, checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        
        # CONTENT ENCODER
        self.content_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1025, out_channels=1025, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=1025),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=1025, out_channels=1025, kernel_size=3, stride=1, padding=1),
            ResidualWrapper(module=nn.Sequential(
                nn.BatchNorm1d(num_features=1025),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv1d(in_channels=1025, out_channels=1025, kernel_size=3, stride=1, padding=1),
            )),
            nn.BatchNorm1d(1025)
        )

        # STYLE ENCODER
        self.style_encoder_1d = nn.Sequential(
            nn.Conv1d(in_channels=1025, out_channels=1025, kernel_size=4, stride=2),
            ResidualWrapper(module=nn.Sequential(
                nn.BatchNorm1d(num_features=1025),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv1d(in_channels=1025,out_channels=1025, kernel_size=1, stride=1)
            )),
            nn.BatchNorm1d(num_features=1025),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.style_encoder_rnn = nn.GRU(batch_first=True, input_size=1025, hidden_size=1025)
        self.style_encoder_0d = nn.Sequential()
        
        # FLOW
        # calculate output dims
        batch_size, in_channels, in_length = input_dims
        out_channels = int(in_channels * 2**(n_levels+1) / 2**n_levels)  # each Squeeze results in 2x in_channels (cf RealNVP section 3.6); each Split in 1/2x in_channels
        # self.output_dims = out_channels, out_HW, out_HW

        # network layers cf Glow figure 2b: (Squeeze -> FlowStep x depth -> Split) x n_levels -> Squeeze -> FlowStep x depth
        self.flowlevels = nn.ModuleList([FlowLevel(in_channels * 2**i, depth, checkpoint_grads) for i in range(n_levels)])
        self.squeeze = Squeeze()
        self.flowstep = FlowSequential(*[FlowStep(out_channels) for _ in range(depth)], checkpoint_grads=checkpoint_grads)

        # gaussianize the final z output; initialize to identity
        self.gaussianize = Gaussianize(out_channels)

        # base distribution of the flow
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

    def forward(self, input_c, input_s, length_c, length_s, return_losses=False):
        # Encode Content and Style
        encoded_c = self.encode_content(input_c)
        encoded_s = self.encode_style(input_s, length_s)
        decoded = self.decode(input_c, encoded_c, encoded_s, length=length_c, max_length=input_c.shape[2])
        return decoded

    def encode_content(self, input):
        encoded = self.content_encoder(input)
        return encoded

    def encode_style(self, input, length):
        encoded = self.style_encoder_1d(input)

        # Mask positions corresponding to padding
        length = torch.div(length,(input.shape[2] / encoded.shape[2]), rounding_mode = 'trunc' ).to(torch.int)
        mask = (torch.arange(encoded.shape[2], device=encoded.device) < length[:, None])[:, None, :]
        encoded *= mask

        # Pass throgh RNN
        encoded = encoded.transpose(1, 2)
        encoded = nn.utils.rnn.pack_padded_sequence(
            encoded, length.clamp(min=1),
            batch_first=True, enforce_sorted=False)
        _, encoded = self.style_encoder_rnn(encoded)

        # Get rid of layer dimension
        encoded = encoded.transpose(0, 1).reshape(input.shape[0], -1)
        encoded = encoded.reshape(encoded.shape[0], -1)
        encoded = self.style_encoder_0d(encoded)
        return encoded

    def decode(self, input_c, encoded_c, encoded_s, length=None, max_length=None):
        encoded_s = encoded_s[:, :, None]          
        conditioning = torch.cat([
                encoded_c,
                encoded_s.expand(-1, -1, encoded_c.shape[-1])
            ], axis=1)
        flow_in = {'melspec':input_c, 'conditioning':conditioning}
        decoded  = self.forward_flow(flow_in)

        # Make sure the output tensor has the same shape as the input tensor
        if max_length is not None or length is not None:
            if max_length is None:
                max_length = length.max()
            decoded = decoded.narrow(-1, 0, max_length)

        # Make sure output lengths are the same as input lengths
        if length is not None:
            mask = (torch.arange(max_length, device=decoded.device) < length[:, None])[:, None, :]
            decoded *= mask

        return decoded

    def forward_flow(self, x):
        '''
        sum_logdets = 0
        zs = []
        for m in self.flowlevels:
            x = m(x)
            sum_logdets = sum_logdets + x['logdet']
            zs.append(z)
        x = self.squeeze(x)
        z, logdet = self.flowstep(x)
        sum_logdets = sum_logdets + logdet
        '''
        sum_logdets = 0
        zs = []
        
        for m in self.flowlevels:
            x = m(x)
            sum_logdets = sum_logdets + x['logdet']
            # zs.append(z)
        '''
        x = self.squeeze(x)
        z, logdet = self.flowstep(x)
        sum_logdets = sum_logdets + logdet

        # gaussianize the final z
        z, logdet = self.gaussianize(torch.zeros_like(z), z)
        sum_logdets = sum_logdets + logdet
        zs.append(z)
        return zs, sum_logdets
        '''

    def inverse_flow(self, zs=None, batch_size=None, z_std=1.):
        if zs is None:  # if no random numbers are passed, generate new from the base distribution
            assert batch_size is not None, 'Must either specify batch_size or pass a batch of z random numbers.'
            zs = [z_std * self.base_dist.sample((batch_size, *self.output_dims)).squeeze()]
        # pass through inverse flow
        z, sum_logdets = self.gaussianize.inverse(torch.zeros_like(zs[-1]), zs[-1])
        x, logdet = self.flowstep.inverse(z)
        sum_logdets = sum_logdets + logdet
        x = self.squeeze.inverse(x)
        for i, m in enumerate(reversed(self.flowlevels)):
            z = z_std * (self.base_dist.sample(x.shape).squeeze() if len(zs)==1 else zs[-i-2])  # if no z's are passed, generate new random numbers from the base dist
            x, logdet = m.inverse(x, z)
            sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def log_prob(self, x, bits_per_pixel=False):
        zs, logdet = self.forward(x)
        log_prob = sum(self.base_dist.log_prob(z).sum([1,2,3]) for z in zs) + logdet
        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel())
        return log_prob

class ResidualWrapper(nn.Module):
    """Wrapper for adding a skip connection around a module."""

    def __init__(self, module=None):
        super().__init__()
        if module is not None:
            self.module = module

    def forward(self, input):
        output = self.module(input)
        if output.shape != input.shape:
            raise RuntimeError(f'Expected output to have shape {input.shape}, got {output.shape}')
        return output + input

# --------------------
# Train and evaluate
# --------------------

@torch.no_grad()
def data_dependent_init(model, args):
    # set up an iterator with batch size = batch_size_init and run through model
    dataloader = fetch_dataloader(args, train=True, data_dependent_init=True)
    model(next(iter(dataloader))[0].requires_grad_(True if args.checkpoint_grads else False).to(args.device))
    del dataloader
    return True

def train_epoch(model, dataloader, optimizer, writer, epoch, args):
    model.train()

    tic = time.time()
    for i, (x,y) in enumerate(dataloader):
        args.step += args.world_size
        # warmup learning rate
        if epoch <= args.n_epochs_warmup:
            optimizer.param_groups[0]['lr'] = args.lr * min(1, args.step / (len(dataloader) * args.world_size * args.n_epochs_warmup))

        x = x.requires_grad_(True if args.checkpoint_grads else False).to(args.device)  # requires_grad needed for checkpointing

        loss = - model.log_prob(x, bits_per_pixel=True).mean(0)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

        optimizer.step()

        # report stats
        if i % args.log_interval == 0:
            # compute KL divergence between base and each of the z's that the model produces
            with torch.no_grad():
                zs, _ = model(x)
                kls = [D.kl.kl_divergence(D.Normal(z.mean(), z.std()), model.base_dist) for z in zs]

            # write stats
            if args.on_main_process:
                et = time.time() - tic              # elapsed time
                tt = len(dataloader) * et / (i+1)   # total time per epoch
                print('Epoch: [{}/{}][{}/{}]\tStep: {}\tTime: elapsed {:.0f}m{:02.0f}s / total {:.0f}m{:02.0f}s\tLoss {:.4f}\t'.format(
                      epoch, args.start_epoch + args.n_epochs, i+1, len(dataloader), args.step, et//60, et%60, tt//60, tt%60, loss.item()))
                # update writer
                for j, kl in enumerate(kls):
                    writer.add_scalar('kl_level_{}'.format(j), kl.item(), args.step)
                writer.add_scalar('train_bits_x', loss.item(), args.step)

        # save and generate
        if i % args.save_interval == 0:
            # generate samples
            samples = generate(model, n_samples=4, z_stds=[0., 0.25, 0.7, 1.0])
            images = make_grid(samples.cpu(), nrow=4, pad_value=1)

            # write stats and save checkpoints
            if args.on_main_process:
                save_image(images, os.path.join(args.output_dir, 'generated_sample_{}.png'.format(args.step)))

                # save training checkpoint
                torch.save({'epoch': epoch,
                            'global_step': args.step,
                            'state_dict': model.state_dict()},
                            os.path.join(args.output_dir, 'checkpoint.pt'))
                torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))



@torch.no_grad()
def evaluate(model, dataloader, args):
    model.eval()
    print('Evaluating ...', end='\r')

    logprobs = []
    for x,y in dataloader:
        x = x.to(args.device)
        logprobs.append(model.log_prob(x, bits_per_pixel=True))
    logprobs = torch.cat(logprobs, dim=0).to(args.device)
    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.std(0) / math.sqrt(len(dataloader.dataset))
    return logprob_mean, logprob_std

@torch.no_grad()
def generate(model, n_samples, z_stds):
    model.eval()
    print('Generating ...', end='\r')

    samples = []
    for z_std in z_stds:
        sample, _ = model.inverse(batch_size=n_samples, z_std=z_std)
        log_probs = model.log_prob(sample, bits_per_pixel=True)
        samples.append(sample[log_probs.argsort().flip(0)])  # sort by log_prob; flip high (left) to low (right)
    return torch.cat(samples,0)

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, writer, args):
    global best_eval_logprob

    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        train_epoch(model, train_dataloader, optimizer, writer, epoch, args)

        # evaluate
        if False:#epoch % args.eval_interval == 0:
            eval_logprob_mean, eval_logprob_std = evaluate(model, test_dataloader, args)
            print('Evaluate at epoch {}: bits_x = {:.3f} +/- {:.3f}'.format(epoch, eval_logprob_mean, eval_logprob_std))

            # save best state
            if args.on_main_process and eval_logprob_mean > best_eval_logprob:
                best_eval_logprob = eval_logprob_mean
                torch.save({'epoch': epoch,
                            'global_step': args.step,
                            'state_dict': model.state_dict()},
                            os.path.join(args.output_dir, 'best_model_checkpoint.pt'))


# --------------------
# Visualizations
# --------------------

def encode_dataset(model, dataloader):
    model.eval()

    zs = []
    attrs = []
    for i, (x,y) in enumerate(dataloader):
        print('Encoding [{}/{}]'.format(i+1, len(dataloader)), end='\r')
        x = x.to(args.device)
        zs_i, _ = model(x)
        zs.append(torch.cat([z.flatten(1) for z in zs_i], dim=1))
        attrs.append(y)

    zs = torch.cat(zs, dim=0)
    attrs = torch.cat(attrs, dim=0)
    print('Encoding completed.')
    return zs, attrs

def compute_dz(zs, attrs, idx):
    """ for a given attribute idx, compute the mean for all encoded z's corresponding to the positive and negative attribute """
    z_pos = [zs[i] for i in range(len(zs)) if attrs[i][idx] == +1]
    z_neg = [zs[i] for i in range(len(zs)) if attrs[i][idx] == -1]
    # dz = z_pos - z_neg; where z_pos is mean of all encoded datapoints where attr is present;
    return torch.stack(z_pos).mean(0) - torch.stack(z_neg).mean(0)   # out tensor of shape (flattened zs dim,)

def get_manipulators(zs, attrs):
    """ compute dz (= z_pos - z_neg) for each attribute """
    print('Extracting manipulators...', end=' ')
    dzs = 1.6 * torch.stack([compute_dz(zs, attrs, i) for i in range(attrs.shape[1])], dim=0)  # compute dz for each attribute official code multiplies by 1.6 scalar here
    print('Completed.')
    return dzs  # out (n_attributes, flattened zs dim)

def manipulate(model, z, dz, z_std, alpha):
    # 1. record incoming shapes
    z_dims   = [z_.squeeze().shape   for z_ in z]
    z_numels = [z_.numel() for z_ in z]
    # 2. flatten z into a vector and manipulate by alpha in the direction of dz
    z = torch.cat([z_.flatten(1) for z_ in z], dim=1).to(dz.device)
    z = z + dz * torch.tensor(alpha).float().view(-1,1).to(dz.device)  # out (n_alphas, flattened zs dim)
    # 3. reshape back to z shapes from each level of the model
    zs = [z_.view((len(alpha), *dim)) for z_, dim in zip(z.split(z_numels, dim=1), z_dims)]
    # 4. decode
    return model.inverse(zs, z_std=z_std)[0]

def load_manipulators(model, args):
    # construct dataloader with limited number of images
    args.mini_data_size = 30000
    # load z manipulators for each attribute
    if os.path.exists(os.path.join(args.output_dir, 'z_manipulate.pt')):
        z_manipulate = torch.load(os.path.join(args.output_dir, 'z_manipulate.pt'), map_location=args.device)
    else:
        # encode dataset, compute manipulators, store zs, attributes, and dzs
        dataloader = fetch_dataloader(args, train=True)
        zs, attrs = encode_dataset(model, dataloader)
        z_manipulate = get_manipulators(zs, attrs)
        torch.save(zs, os.path.join(args.output_dir, 'zs.pt'))
        torch.save(attrs, os.path.join(args.output_dir, 'attrs.pt'))
        torch.save(z_manipulate, os.path.join(args.output_dir, 'z_manipulate.pt'))
    return z_manipulate

@torch.no_grad()
def visualize(model, args, attrs=None, alphas=None, img_path=None, n_examples=1):
    """ manipulate an input image along a given attribute """
    dataset = fetch_dataloader(args, train=False).dataset  # pull the dataset to access transforms and attrs
    # if no attrs passed, manipulate all of them
    if not attrs:
        attrs = list(range(len(dataset.attr_names)))
    # if image is passed, manipulate only the image
    if img_path:
        from PIL import Image
        img = Image.open(img_path)
        x = dataset.transform(img)  # transform image to tensor and encode
    else:  # take first n_examples from the dataset
        x, _ = dataset[0]
    z, _ = model(x.unsqueeze(0).to(args.device))
    # get manipulors
    z_manipulate = load_manipulators(model, args)
    # decode the varied attributes
    dec_x =[]
    for attr_idx in attrs:
        dec_x.append(manipulate(model, z, z_manipulate[attr_idx].unsqueeze(0), args.z_std, alphas))
    return torch.stack(dec_x).cpu()


# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()
    args.step = 0  # global step
    args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else os.path.join(args.output_dir, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    writer = None  # init as None in case of multiprocessing; only main process performs write ops

    # setup device and distributed training
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device('cuda:{}'.format(args.local_rank))

        # initialize
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        # compute total world size (used to keep track of global step)
        args.world_size = int(os.environ['WORLD_SIZE'])  # torch.distributed.launch sets this to nproc_per_node * nnodes
    else:
        if torch.cuda.is_available(): args.local_rank = 0
        args.device = torch.device('cuda:{}'.format(args.local_rank) if args.local_rank is not None else 'cpu')

    # write ops only when on_main_process
    # NOTE: local_rank unique only to the machine; only 1 process on each node is on_main_process;
    #       if shared file system, args.local_rank below should be replaced by global rank e.g. torch.distributed.get_rank()
    args.on_main_process = (args.distributed and args.local_rank == 0) or not args.distributed

    # setup seed
    if args.seed:
        torch.manual_seed(args.seed)
        if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # load data; sets args.input_dims needed for setting up the model
    train_dataloader = fetch_dataloader(args, train=True)
    test_dataloader = fetch_dataloader(args, train=False)

    
    # load model
    model = GlowTT( args.depth, args.n_levels, args.input_dims, args.checkpoint_grads).to(args.device)
    if args.distributed:
        # NOTE: DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        # for compatibility of saving/loading models, wrap non-distributed cpu/gpu model as well;
        # ie state dict is based on model.module.layer keys, which now match between training distributed and running then locally
        model = torch.nn.parallel.DataParallel(model)
    # DataParalle and DistributedDataParallel are wrappers around the model; expose functions of the model directly
    model.base_dist = model.module.base_dist
    model.log_prob = model.module.log_prob
    model.inverse = model.module.inverse_flow

    for (input_c, length_c), (input_s, length_s) in train_dataloader:
        input_c, length_c, input_s, length_s = (x.to(args.device) for x in (input_c, length_c, input_s, length_s))
        _, losses = model(input_c, input_s, length_c, length_s, return_losses=True)
        break

    '''
    # load optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # load checkpoint if provided
    if args.restore_file:
        model_checkpoint = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location=args.device))
        args.start_epoch = model_checkpoint['epoch']
        args.step = model_checkpoint['global_step']

    # setup writer and outputs
    if args.on_main_process:
        writer = SummaryWriter(log_dir = args.output_dir)

        # save settings
        config = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__)) + \
                 'Num trainable params: {:,.0f}\n\n'.format(sum(p.numel() for p in model.parameters())) + \
                 'Model:\n{}'.format(model)
        config_path = os.path.join(args.output_dir, 'config.txt')
        writer.add_text('model_config', config)
        if not os.path.exists(config_path):
            with open(config_path, 'a') as f:
                print(config, file=f)
    
    if args.train:
        # run data dependent init and train
        data_dependent_init(model, args)
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, writer, args)
    
    if args.evaluate:
        logprob_mean, logprob_std = evaluate(model, test_dataloader, args)
        print('Evaluate: bits_x = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std))

    if args.generate:
        n_samples = 4
        z_std = [0., 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if not args.z_std else n_samples * [args.z_std]
        samples = generate(model, n_samples, z_std)
        images = make_grid(samples.cpu(), nrow=n_samples, pad_value=1)
        save_image(images, os.path.join(args.output_dir,
                                        'generated_samples_at_z_std_{}.png'.format('range' if args.z_std is None else args.z_std)))

    if args.visualize:
        if not args.z_std: args.z_std = 0.6
        if not args.vis_alphas: args.vis_alphas = [-2,-1,0,1,2]
        dec_x = visualize(model, args, args.vis_attrs, args.vis_alphas, args.vis_img)   # output (n_attr, n_alpha, 3, H, W)
        filename = 'manipulated_sample' if not args.vis_img else \
                   'manipulated_img_{}'.format(os.path.basename(args.vis_img).split('.')[0])
        if args.vis_attrs:
            filename += '_attr_' + ','.join(map(str, args.vis_attrs))
        save_image(dec_x.view(-1, *args.input_dims), os.path.join(args.output_dir, filename + '.png'), nrow=dec_x.shape[1])

    if args.on_main_process:
        writer.close()
    '''