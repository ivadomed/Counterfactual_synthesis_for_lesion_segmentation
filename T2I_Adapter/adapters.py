
"Largely taken and adapted from https://github.com/TencentARC/T2I-Adapter"

import torch
import math
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import os


##### Functions used in the training loop #####
def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def noop(*args, **kwargs):
    pass

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images

def re_orient(image):
    """ Take a 2D of a spinal coord as input oriented as expected by the model 
    and return the image oriented as expected by the conventions
    """
    return np.flip(np.rot90(image,1),1)

##### Functions used Network architecture #####

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=3, out_channels=None, padding=1, is_input=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if is_input or dims!=3:
            stride = 2
        else:
            stride = (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ZeroConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ZeroConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # Initialize weights to zero
        init.constant_(self.conv.weight, 0)
        # If the layer includes a bias, initialize it to zero (optional)
        if bias:
            init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ResnetBlock(nn.Module):
    """
    Residual block used in a ResNet architecture.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        down (bool): Whether to downsample the input.
        ksize (int, optional): Kernel size for convolutional layers. Defaults to 3.
        sk (bool, optional): Whether to use skip connection. Defaults to False.
        use_conv (bool, optional): Whether to use convolutional downsampling. Defaults to True.
    """

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True, zero_conv=False):
        super().__init__()
        ps = ksize // 2
        self.zero_conv = zero_conv
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv3d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv3d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv3d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv3d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)
        # create a zero convolutional layer, this way, the T2I has no impact on output at the begging of training
        self.zero_conv = ZeroConv3d(out_c, out_c, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            h = h + self.skep(x)
        else:
            h = self.zero_conv(h + x)
        if self.zero_conv:
            h = self.zero_conv(h)
        return h


class Adapter_Medical_Diffusion(nn.Module):
    """
    This object is a T2I_adapter tailored for medical diffusion network. 
    It takes as an input an image control and outputs a list of features to be added to the 4 diffusion UNet encoder layers outputs.
    """

    def __init__(self, channels=[128, 256, 512, 1024], nums_rb=3, cin=1, ksize=3, sk=True, use_conv=True, zero_conv=False):
        super(Adapter_Medical_Diffusion, self).__init__()

        # Load the 


        self.unshuffle = nn.PixelUnshuffle(16)
        self.input_down = Downsample(cin, use_conv=use_conv, is_input=True)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i >= 1) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv, zero_conv=zero_conv))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv, zero_conv=zero_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv3d(cin, channels[0], 3, 1, 1)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x):
        # unshuffle
        #x = self.unshuffle(x)
        x = self.input_down(x)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)

        return features

class T2I_Trainer(object):
    def __init__(self, T2I_model=None, T2I_derivative_name=None, diffusion_model=None, amp=False, dataset=None, batch_size=1, save_and_sample_every=5000, train_lr=1e-5, train_num_steps=50000, gradient_accumulate_every=2, results_folder='./results', num_workers=20):
        self.T2I_model = T2I_model.cuda()
        self.T2I_derivative_name = T2I_derivative_name
        self.diffusion = diffusion_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.save_and_sample_every = save_and_sample_every
        self.train_lr = train_lr
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every
        self.results_folder = results_folder
        self.step = 0
        self.amp=amp
        self.scaler = GradScaler(enabled=amp)
        self.opt = Adam(T2I_model.parameters(), lr=train_lr)

        dl = DataLoader(self.dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers)
        self.len_dataloader = len(dl)
        self.dl = cycle(dl)
    
    def load(self, T2I_pt):
        """
        Load a pretrained model.
        """
        self.T2I_model.load_state_dict(torch.load(T2I_pt))
        print('loaded model')

    def inference_to_png(self, inference, T2I_image):
        """
        Save the output of the diffusion model and the T2I image to a png file with the T2I_image guidance as overlay
        """

        # plot the input, output with and without T2I?, add as well the T2I input as a red overlay with 0.5 alpha
        T2I_image = np.ma.masked_equal(T2I_image.cpu().numpy(), 0)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(re_orient(inference[0,0,16].cpu().numpy()), cmap='gray', vmin=-1, vmax=1)
        ax.imshow(re_orient(T2I_image[0,0,16]), cmap='Reds', alpha=0.5,  vmin=0, vmax=1)
        ax.axis('off')

        # save the figure
        fig.savefig(f'{self.results_folder}/{self.step}.png')


    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                sample = next(self.dl)
                image = sample['data'].cuda()
                T2I_image = sample[self.T2I_derivative_name].cuda()
                with autocast(enabled=self.amp):
                    T2I_features = self.T2I_model(T2I_image)
                    loss = self.diffusion(
                        image,
                        T2I_features=T2I_features,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask
                    )

                    self.scaler.scale(
                        loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}


            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()


            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                if not os.path.exists(self.results_folder):
                    os.makedirs(self.results_folder)
                # save model
                torch.save(self.T2I_model.state_dict(), f'{self.results_folder}/T2I_model_{self.step}.pt')
                # run an inference
                with torch.no_grad():
                    self.T2I_model.eval()
                    sample = next(self.dl)
                    T2I_image = sample[self.T2I_derivative_name].cuda()
                    T2I_features = self.T2I_model(T2I_image)
                    image = sample['data'].cuda()
                    inference = self.diffusion.sample(batch_size=1, T2I_features = T2I_features)
                    self.inference_to_png(inference, T2I_image)
                    self.T2I_model.train() 

            log_fn(log)
            self.step += 1

        torch.save(self.T2I_model.state_dict(), f'{self.results_folder}/T2I_model_last.pt')
        print('training completed')
        

"""
# create a Adapter_Medical_Diffusion model
model = Adapter_Medical_Diffusion(sk=True)

# create a random input tensor
x = torch.randn(1, 1, 32, 256, 256)

# forward pass
output = model(x)
for i in range (len(output)):
    print("output " + str(i) + " : " + str(output[i].shape))
    # print max of each output
    print("max of output " + str(i) + " : " + str(output[i].max().item()))
"""

"""
result :
output 0 : torch.Size([3, 128, 16, 128, 128])
output 1 : torch.Size([3, 256, 16, 64, 64])
output 2 : torch.Size([3, 512, 16, 32, 32])
output 3 : torch.Size([3, 1024, 16, 16, 16])
"""
