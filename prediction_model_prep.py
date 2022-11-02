import numpy as np
import torch
from torch import nn
from nnunet.network_architecture.generic_UNet import Generic_UNet_predict
from nnunet.network_architecture.initialization import InitWeights_He
import torchio as tio
import glob
from sys import argv


def get_pretrained_model(pth, dev):
    """
    pth: path to pretrained UNet model to get weights
    dev: where to load the pt weights (str)
    """
    dev = torch.device(dev)

    pt_weights = torch.load(pth, map_location=torch.device(dev))

    num_input_channels = 1
    base_num_features = 32
    num_classes = 2
    net_num_pool_op_kernel_sizes = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
    conv_per_stage = 2
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    net_conv_kernel_sizes = [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],[3, 3, 3], [3, 3, 3]]

    # deep supervision is used during training the segmentation network. If True, outputs of all decoding blocks, exlcuding
    # the two lowest resolutions, will be used in an auxiliary loss functions. When False, only final output is returned
    deep_supervision = True

    network = Generic_UNet_predict(num_input_channels, base_num_features, num_classes,
                                len(net_num_pool_op_kernel_sizes),
                                conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                dropout_op_kwargs,
                                net_nonlin, net_nonlin_kwargs, deep_supervision, False, lambda x: x, InitWeights_He(1e-2),
                                net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

    # put model on same device as the pt weights model
    network.to(dev)
    network.load_state_dict(pt_weights['state_dict'])

    return network

# add data prep + getting cat
def data_prep(pth, seg_net, dev):
    # """
    # pth: path to .npz file
    # seg_net: loaded UNet torch model
    # dev: device to run prep (make sure it's the same dev as the seg_net)
    # """
    dev = torch.device(dev)
    # load file
    dat = np.load(pth)
    # extract volume data, convert to correct format, create subject
    vol = dat['data'][0,:,:,:]
    vol = torch.tensor(vol).unsqueeze(0)
    sub = tio.Subject(ct = tio.ScalarImage(tensor=vol))

    # sample all patches, run through seg model to get outputs of interest
    sampler = tio.GridSampler(subject=sub,patch_size=(28, 256, 256))
    all_patches = torch.empty(0)
    seg_net.eval()
    for i, patch in enumerate(sampler):
        # need to run one patch at a time due to GPU memory limitations... 
    	with torch.no_grad():
            _, skips, _, _ = seg_net(patch.ct.data.unsqueeze(0).to(dev))
            pred_in = torch.empty(0)
            for s in skips:
                p = torch.mean(s, axis = [2,3,4])
                pred_in = torch.cat((pred_in, p.detach().cpu()), axis =-1)
            all_patches = torch.cat((all_patches, pred_in), dim=0)

    return all_patches

# path to pretrained segmentation model weights
pt_pth = str(argv[1])
# path to data dir
data_pth = str(argv[2])
# path to write processed data
out_pth = str(argv[3])
# device argument 
dev = str(argv[4])

# load seg model
pt_seg_model = get_pretrained_model(pt_pth, dev)

for n in glob.glob(data_pth+"*.npz"):
	dat = data_prep(n, pt_seg_model, dev)
	torch.save(dat, out_pth+n.split("/")[-1][:-4]+".pt")











