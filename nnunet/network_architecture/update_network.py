import torch
import torch.nn as nn



class ConvBlock(nn.Module):
	"""
	Convolution, normalization, non-linearity
	"""

	def __init__(self, in_channels, out_channels, conv_op = nn.Conv3d
				 conv_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True},
				 norm_op = nn.InstanceNorm3d, 
				 norm_kwargs=None, nonlin_op=nn.LeakyReLU, nonlin_kwargs=None)
		super(ConvBlock, self).__init__()

		if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv = conv
        nn.Conv3d(in_channels, out_channels, **conv_kwargs)

            