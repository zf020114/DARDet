import math

import torch.nn as nn
from mmcv.cnn import ConvModule,build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import  kaiming_init
from mmdet.models.builder import NECKS
from mmcv.ops import DeformConv2dPack

@NECKS.register_module()
class ShortcutNeck(BaseModule):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.

    Args:
         in_channel (int): Number of input channels.
         num_deconv_filters (tuple[int]): Number of filters per stage.
         num_deconv_kernels (tuple[int]): Number of kernels per stage.
         use_dcn (bool): If True, use DCNv2. Default: True.
         init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                inplanes=(128, 256, 512, 1024),
                planes=(256, 128, 64),
                shortcut_kernel=3,
                shortcut_cfg=(1, 2, 3),
                norm_cfg=dict(type='BN', requires_grad=True),
                use_dcn=True,
                init_cfg=None):
        super(ShortcutNeck, self).__init__(init_cfg)
        # assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        shortcut_num = min(len(inplanes) - 1, len(planes))
        assert shortcut_num == len(shortcut_cfg)
        # self.in_channel = in_channel
        # self.deconv_layers = self._make_deconv_layer(num_deconv_filters,
        #                                              num_deconv_kernels)

        # repeat upsampling n times. 32x to 4x by default.
        self.deconv_layers = nn.ModuleList([
            self.build_upsample(inplanes[-1], planes[0], norm_cfg=norm_cfg),
            self.build_upsample(planes[0], planes[1], norm_cfg=norm_cfg)
        ])
        for i in range(2, len(planes)):
            self.deconv_layers.append(
                self.build_upsample(planes[i - 1], planes[i], norm_cfg=norm_cfg))

        padding = (shortcut_kernel - 1) // 2
        self.shortcut_layers = self.build_shortcut(
            inplanes[:-1][::-1][:shortcut_num], planes[:shortcut_num], shortcut_cfg,
            kernel_size=shortcut_kernel, padding=padding)


    def build_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       kernel_size=3,
                       padding=1):
        assert len(inplanes) == len(planes) == len(shortcut_cfg)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num) in zip(
                inplanes, planes, shortcut_cfg):
            assert layer_num > 0
            layer = ShortcutConv2d(
                inp, outp, [kernel_size] * layer_num, [padding] * layer_num)
            shortcut_layers.append(layer)
        return shortcut_layers

    def build_upsample(self, inplanes, planes, norm_cfg=None):
        mdcn = DeformConv2dPack(inplanes, planes, 3, stride=1,
                                       padding=1, dilation=1, deformable_groups=1)
        up = nn.UpsamplingBilinear2d(scale_factor=2)
        layers = []
        layers.append(mdcn)
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, planes)[1])
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)

        return nn.Sequential(*layers)

    def init_weights(self):
        for _, m in self.shortcut_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = feats[-1]
        for i, upsample_layer in enumerate(self.deconv_layers):
            x = upsample_layer(x)
            if i < len(self.shortcut_layers):
                shortcut = self.shortcut_layers[i](feats[-i - 2])
                x = x + shortcut
        return x


class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y



    # def _make_deconv_layer(self, num_deconv_filters, num_deconv_kernels):
    #     """use deconv layers to upsample backbone's output."""
    #     layers = []
    #     for i in range(len(num_deconv_filters)):
    #         feat_channel = num_deconv_filters[i]
    #         conv_module = ConvModule(
    #             self.in_channel,
    #             feat_channel,
    #             3,
    #             padding=1,
    #             conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
    #             norm_cfg=dict(type='BN'))
    #         layers.append(conv_module)
    #         upsample_module = ConvModule(
    #             feat_channel,
    #             feat_channel,
    #             num_deconv_kernels[i],
    #             stride=2,
    #             padding=1,
    #             conv_cfg=dict(type='deconv'),
    #             norm_cfg=dict(type='BN'))
    #         layers.append(upsample_module)
    #         self.in_channel = feat_channel

    #     return nn.Sequential(*layers)

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.ConvTranspose2d):
    #             # In order to be consistent with the source code,
    #             # reset the ConvTranspose2d initialization parameters
    #             m.reset_parameters()
    #             # Simulated bilinear upsampling kernel
    #             w = m.weight.data
    #             f = math.ceil(w.size(2) / 2)
    #             c = (2 * f - 1 - f % 2) / (2. * f)
    #             for i in range(w.size(2)):
    #                 for j in range(w.size(3)):
    #                     w[0, 0, i, j] = \
    #                         (1 - math.fabs(i / f - c)) * (
    #                                 1 - math.fabs(j / f - c))
    #             for c in range(1, w.size(0)):
    #                 w[c, 0, :, :] = w[0, 0, :, :]
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         # self.use_dcn is False
    #         elif not self.use_dcn and isinstance(m, nn.Conv2d):
    #             # In order to be consistent with the source code,
    #             # reset the Conv2d initialization parameters
    #             m.reset_parameters()

    # @auto_fp16()
    # def forward(self, inputs):
    #     assert isinstance(inputs, (list, tuple))
    #     outs = self.deconv_layers(inputs[-1])
    #     return outs,
