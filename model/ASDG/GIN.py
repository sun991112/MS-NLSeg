import torch.nn as nn
import torch
import numpy as np
import functools
import sys
sys.path.append('/home/model')
from torch.autograd import Function
from torch.nn import init
#from model.unet import UNet
# from model.loss import FixedLoss
from unet import UNet
import torch.nn.functional as F
from loss import FixedLoss,ASDGPatchNCELoss
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
        self.style_dim = style_dim

    def forward(self, x):
        s = torch.randn(x.shape[0],self.style_dim).cuda()
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class NormalNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            mu = weight_mat.mean()
            std = weight_mat.std()
            # print(mu,std)
        weight_sn = (weight-mu) / std

        return weight_sn

    @staticmethod
    def apply(module, name):
        fn = NormalNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        module.register_buffer(name, weight)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn = self.compute_weight(module)
        setattr(module, self.name, weight_sn)


def spectral_norm(module, name='weight'):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        NormalNorm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight, gain)

    return spectral_norm(module)

class GIN(nn.Module):
    def __init__(self):
        super(GIN, self).__init__()
        ch = 2
        self.net1 = nn.Sequential(
            nn.Conv2d(3,ch,3,padding=1),
            (AdaIN(2,ch) ),
            nn.LeakyReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            (AdaIN(2,ch) ),
            nn.LeakyReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            (AdaIN(2,ch) ),
            nn.LeakyReLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            (AdaIN(2,ch) ),
            nn.LeakyReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            (AdaIN(2,ch) ),
            nn.LeakyReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            (AdaIN(2,ch) ),
            nn.LeakyReLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

        self.__initialize_weights()
        self.apply(spectral_init)

        # print(self.net1, self.net2)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1.0)

    def normalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

                weight = m.weight.data*1.0
                m.weight.data = (m.weight.data - weight.mean())/weight.std()

    def forward(self, x):

        out1 = self.net1(x)
        out2 = self.net2(x)

        return out1, out2
def mix_out(x, out1, out2):

    alpha = torch.rand(2,out1.shape[0],1,1,1).cuda()

    out1 = out1 * alpha[0] + (1 - alpha[0]) * x
    out2 = out2 * alpha[1] + (1 - alpha[1]) * x

    out1 = out1 * ((torch.square(x).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True)).sqrt() /
                   (torch.square(out1.detach()).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True).sum(dim=1,
                                                                                                      keepdim=True)).sqrt()).detach()

    out2 = out2 * ((torch.square(x).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True)).sqrt() /
                   (torch.square(out2.detach()).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True).sum(dim=1,
                                                                                                      keepdim=True)).sqrt()).detach()
    return out1, out2

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=['cuda:0']):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids
class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=True):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out
class ASDG_model(nn.Module):
    def __init__(self):
        super(ASDG_model, self).__init__()
        self.model=UNet()
        self.GIN=GIN()
        self.netF=PatchSampleF(use_mlp=True, init_gain=0.02)
        self.netG=ResnetGenerator(input_nc=3, output_nc=3, ngf=32, norm_layer=nn.InstanceNorm2d,
                          use_dropout=False, no_antialias=False, no_antialias_up=False, n_blocks=6)

    def foward1(self,input_image):
        aug_img1_, aug_img2_ = self.GIN(input_image)
        aug_img1, aug_img2 = mix_out(input_image, aug_img1_, aug_img2_)
        predict1 = self.model(aug_img1.detach())
        predict2 = self.model(aug_img2.detach())
        return predict1, predict2,aug_img1_, aug_img2_,aug_img1,aug_img2

    def foward2(self,src,aug_img1_,aug_img1,aug_img2):
        feat_q = self.netG(aug_img1_, [0,4,8,12,16], encode_only=True)
        feat_k = self.netG(src, [0,4,8,12,16], encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, 256, None)
        feat_q_pool, _ = self.netF(feat_q, 256, sample_ids)
        bs = src.shape[0]

        total_nce_loss = 0.0
        for f_q, f_k, nce_layer in zip(feat_q_pool, feat_k_pool, [0,4,8,12,16]):
            loss = ASDGPatchNCELoss()(f_q, f_k, bs) * 1.0

            total_nce_loss += loss.mean()
        miloss=total_nce_loss /5

        with torch.no_grad():
            predict1=self.model(aug_img1)
            predict2=self.model(aug_img2)

        return predict1, predict2,miloss
    def test_forward(self,img):
        return self.model(img)

def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x
if __name__ == '__main__':
    img=torch.randn(8,3,224,224).to('cuda')
    label=torch.randn(8,3,224,224).to('cuda')
    model=ASDG_model().to('cuda')

    # opt.zero_grad()
    # opt_mi.zero_grad()
    predict1, predict2, aug_img1_, aug_img2_,aug_img1,aug_img2=model.foward1(img)
    # l1=(FixedLoss(predict1,label)+FixedLoss(predict2,label))/2 + ASDGKLloss(predict1,predict2)
    # opt.step()
    new_predict1, new_predict2,miloss=model.foward2(img,aug_img1_,aug_img1,aug_img2)
    # l2=-ASDGKLloss(new_predict1,new_predict2)+miloss
    # opt_mi.step()
    predtccccc=model.test_forward(img)
    print(predtccccc.shape)

