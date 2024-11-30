import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import functional as F
import einops
from typing import Callable, Any
import math
from einops import rearrange, repeat


from semseg.models.layers import DropPath
from semseg.models.backbones.vmamba import LayerNorm2d, PatchMerging2D, VSSBlock, Permute, Linear2d, MambaCross
from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from semseg.models.backbones.csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit, SelectiveScanOflex
from semseg.models.backbones.csm_triton import CrossScanTriton, CrossMergeTriton
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from semseg.models.modules.ffm import FeatureRectifyModule as FRM
from semseg.models.modules.ffm import FeatureFusionModule as FFM
from semseg.models.modules.ffm import tempFFM as TFFM

import copy





class MulMamba(nn.Module):
    def __init__(self,
                 model_name="mulmamba",
                 modals: list = ['rgb', 'depth', 'event', 'lidar'],
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 depths=[2, 2, 8, 2],
                 dims=96,
                 # ===================
                 ssm_d_state=1,
                 ssm_ratio=1.0,
                 ssm_rank_ratio=2.0,
                 ssm_dt_rank="auto",
                 ssm_act_layer="silu",
                 ssm_conv=3,
                 ssm_conv_bias=False,
                 ssm_drop_rate=0.0,
                 ssm_init="v0",
                 forward_type="v05_noz",
                 # ===================
                 mlp_ratio=4.0,
                 mlp_act_layer="gelu",
                 mlp_drop_rate=0.0,
                 # ===================
                 drop_path_rate=0.2,
                 patch_norm=True,
                 norm_layer="ln2d",
                 downsample_version="v3",
                 patchembed_version="v2",
                 gmlp=False,
                 use_checkpoint=False,
                 # ===================
                 posembed=False,
                 imgsize=224
                 # ===================

                 ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.channels = dims
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.modals = modals
        self.num_modals = len(self.modals)


        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)
        _make_patch_embed = self._make_patch_embed_v2
        _make_extra_path_embed = self._make_extra_path_embed
        _make_downsample = self._make_downsample_v3

        # RGB_path_embed
        self.rgb_path_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                             channel_first=self.channel_first)
        # extra path embed
        self.extra_path_embed = nn.ModuleList([_make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                             channel_first=self.channel_first) for _ in range(self.num_modals - 1)])


        # RGB block1
        downsample = _make_downsample(
            self.dims[0],
            self.dims[0 + 1],
            norm_layer=norm_layer,
            channel_first=self.channel_first,
        ) if (0 < self.num_layers - 1) else nn.Identity()
        self.rgb_block1 = self._make_layer(
                dim=self.dims[0],
                drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            )

        # extra block1
        downsample = _make_downsample(
            self.dims[0],
            self.dims[0 + 1],
            norm_layer=norm_layer,
            channel_first=self.channel_first,
        ) if (0 < self.num_layers - 1) else nn.Identity()
        if self.num_modals > 1:
            self.extra_block1 = nn.ModuleList()
            for i in range(self.num_modals - 1):
                self.extra_block1.append(self._make_layer(
                    dim=self.dims[0],
                    drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    downsample=downsample,
                    channel_first=self.channel_first,
                    # =================
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    # =================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                ))

            self.down_channel = Down_channelV2(num_modals=self.num_modals, dims=self.dims)


        # RGB block2
        downsample = _make_downsample(
                self.dims[1],
                self.dims[1 + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (1 < self.num_layers - 1) else nn.Identity()
        self.rgb_block2 = self._make_layer(
                dim=self.dims[1],
                drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            )

        # extra block2
        # downsample = _make_downsample(
        #     self.dims[1],
        #     self.dims[1 + 1],
        #     norm_layer=norm_layer,
        #     channel_first=self.channel_first,
        # ) if (1 < self.num_layers - 1) else nn.Identity()
        #
        # self.extra_block2 = self._make_layer(
        #         dim=self.dims[1],
        #         drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
        #         use_checkpoint=use_checkpoint,
        #         norm_layer=norm_layer,
        #         downsample=downsample,
        #         channel_first=self.channel_first,
        #         # =================
        #         ssm_d_state=ssm_d_state,
        #         ssm_ratio=ssm_ratio,
        #         ssm_dt_rank=ssm_dt_rank,
        #         ssm_act_layer=ssm_act_layer,
        #         ssm_conv=ssm_conv,
        #         ssm_conv_bias=ssm_conv_bias,
        #         ssm_drop_rate=ssm_drop_rate,
        #         ssm_init=ssm_init,
        #         forward_type=forward_type,
        #         # =================
        #         mlp_ratio=mlp_ratio,
        #         mlp_act_layer=mlp_act_layer,
        #         mlp_drop_rate=mlp_drop_rate,
        #         gmlp=gmlp,
        #     )

        # RGB block3
        downsample = _make_downsample(
                self.dims[2],
                self.dims[2 + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (2 < self.num_layers - 1) else nn.Identity()
        self.rgb_block3 = self._make_layer(
                dim=self.dims[2],
                drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            )

        # extra block3
        # downsample = _make_downsample(
        #     self.dims[2],
        #     self.dims[2 + 1],
        #     norm_layer=norm_layer,
        #     channel_first=self.channel_first,
        # ) if (2 < self.num_layers - 1) else nn.Identity()
        # self.extra_block3 = self._make_layer(
        #         dim=self.dims[2],
        #         drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
        #         use_checkpoint=use_checkpoint,
        #         norm_layer=norm_layer,
        #         downsample=downsample,
        #         channel_first=self.channel_first,
        #         # =================
        #         ssm_d_state=ssm_d_state,
        #         ssm_ratio=ssm_ratio,
        #         ssm_dt_rank=ssm_dt_rank,
        #         ssm_act_layer=ssm_act_layer,
        #         ssm_conv=ssm_conv,
        #         ssm_conv_bias=ssm_conv_bias,
        #         ssm_drop_rate=ssm_drop_rate,
        #         ssm_init=ssm_init,
        #         forward_type=forward_type,
        #         # =================
        #         mlp_ratio=mlp_ratio,
        #         mlp_act_layer=mlp_act_layer,
        #         mlp_drop_rate=mlp_drop_rate,
        #         gmlp=gmlp,
        #     )

        # RGB block4
        downsample = _make_downsample(
                self.dims[3],
                self.dims[3 + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (3 < self.num_layers - 1) else nn.Identity()
        self.rgb_block4 = self._make_layer(
                dim=self.dims[3],
                drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            )

        # extra block4
        # downsample = _make_downsample(
        #     self.dims[3],
        #     self.dims[3 + 1],
        #     norm_layer=norm_layer,
        #     channel_first=self.channel_first,
        # ) if (3 < self.num_layers - 1) else nn.Identity()
        #
        # self.extra_block4 = self._make_layer(
        #         dim=self.dims[3],
        #         drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
        #         use_checkpoint=use_checkpoint,
        #         norm_layer=norm_layer,
        #         downsample=downsample,
        #         channel_first=self.channel_first,
        #         # =================
        #         ssm_d_state=ssm_d_state,
        #         ssm_ratio=ssm_ratio,
        #         ssm_dt_rank=ssm_dt_rank,
        #         ssm_act_layer=ssm_act_layer,
        #         ssm_conv=ssm_conv,
        #         ssm_conv_bias=ssm_conv_bias,
        #         ssm_drop_rate=ssm_drop_rate,
        #         ssm_init=ssm_init,
        #         forward_type=forward_type,
        #         # =================
        #         mlp_ratio=mlp_ratio,
        #         mlp_act_layer=mlp_act_layer,
        #         mlp_drop_rate=mlp_drop_rate,
        #         gmlp=gmlp,
        #     )

        if self.num_modals > 1:
            self.FRMs = nn.ModuleList([
                FRM(dim=self.dims[0],reduction=1),
                FRM(dim=self.dims[1],reduction=1),
                FRM(dim=self.dims[2],reduction=1),
                FRM(dim=self.dims[3],reduction=1)
            ])


            # self.FFMs = nn.ModuleList([
            #     TFFM(),
            #     TFFM(),
            #     TFFM(),
            #     TFFM()
            # ])

            # num_heads = [1, 2, 4, 8]
            # self.FFMs = nn.ModuleList([
            #     FFM(dim=self.dims[0], reduction=1, num_heads=num_heads[0], norm_layer=nn.BatchNorm2d),
            #     FFM(dim=self.dims[1], reduction=1, num_heads=num_heads[1], norm_layer=nn.BatchNorm2d),
            #     FFM(dim=self.dims[2], reduction=1, num_heads=num_heads[2], norm_layer=nn.BatchNorm2d),
            #     FFM(dim=self.dims[3], reduction=1, num_heads=num_heads[3], norm_layer=nn.BatchNorm2d)])

            self.FFMs = nn.ModuleList([
                CrossMambaFusionBlock(hidden_dim=self.dims[i],
                drop_path=0,
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,) for i in range(4)])

            # self.FFMs = nn.ModuleList([ConvMerge(self.dims[i] * 2, self.dims[i] * 3 // 2, self.dims[i]) for i in range(4)])


        self.num_ber = 0
        self.attention_path = "attention_vis2"
        self.apply(self._init_weights)





    def forward(self, x: list):
        out = []
        x_rgb = x[0]
        if self.num_modals > 1:
            x_ext = x[1:]

        B, C, H, W = x_rgb.shape

        # stage 1
        x_rgb = self.rgb_path_embed(x_rgb)
        x_rgb = self.rgb_block1[0](x_rgb)
        if self.num_modals > 1:
            x_ext = [self.extra_path_embed[i](x_ext[i]) for i in range(self.num_modals - 1)]
            x_ext = [self.extra_block1[i][0](x_ext[i]) for i in range(self.num_modals - 1)]
            x_ext = self.down_channel(x_ext)
            x_rgb, x_ext = self.FRMs[0](x_rgb, x_ext)
            x_fused = self.FFMs[0](x_rgb, x_ext)
            out.append(x_fused)
            x_rgb = self.rgb_block1[1](x_rgb)
            x_ext = self.extra_block1[0][1](x_ext)
        else:
            out.append(x_rgb)
            x_rgb = self.rgb_block1[1](x_rgb)

        # stage 2
        x_rgb = self.rgb_block2[0](x_rgb)
        if self.num_modals > 1:
            x_ext = self.rgb_block2[0](x_ext)
            x_rgb, x_ext = self.FRMs[1](x_rgb, x_ext)
            x_fused = self.FFMs[1](x_rgb, x_ext)
            out.append(x_fused)
            x_rgb = self.rgb_block2[1](x_rgb)
            x_ext = self.rgb_block2[1](x_ext)
        else:
            out.append(x_rgb)
            x_rgb = self.rgb_block2[1](x_rgb)

        # stage 3
        x_rgb = self.rgb_block3[0](x_rgb)

        # x_rgb_ = x_rgb.clone()

        if self.num_modals > 1:
            x_ext = self.rgb_block3[0](x_ext)
            x_rgb, x_ext = self.FRMs[2](x_rgb, x_ext)

            # x_ext_ = x_ext.clone()

            x_fused = self.FFMs[2](x_rgb, x_ext)


            # self.vision_features(x_rgb_, x_ext_, x_fused)


            out.append(x_fused)
            x_rgb = self.rgb_block3[1](x_rgb)
            x_ext = self.rgb_block3[1](x_ext)
        else:
            out.append(x_rgb)
            x_rgb = self.rgb_block3[1](x_rgb)

        # stage 4
        x_rgb = self.rgb_block4[0](x_rgb)
        if self.num_modals > 1:
            x_ext = self.rgb_block4[0](x_ext)
            x_rgb, x_ext = self.FRMs[3](x_rgb, x_ext)
            x_fused = self.FFMs[3](x_rgb, x_ext)
            out.append(x_fused)
        else:
            out.append(x_rgb)

        return out

    @torch.no_grad()
    def vision_features(self, rgb, x_e, x_fused):
        rgb = rgb.squeeze(0)
        x_e = x_e.squeeze(0)
        x_fused = x_fused.squeeze(0)

        rgb = torch.max(rgb, 0)[0].detach().cpu().numpy()
        x_e = torch.max(x_e, 0)[0].detach().cpu().numpy()
        x_fused = torch.max(x_fused, 0)[0].detach().cpu().numpy()

        rgb = self.feature2rgb(rgb)
        x_e = self.feature2rgb(x_e)
        x_fused = self.feature2rgb(x_fused)

        rgb_path = os.path.join(self.attention_path,str(self.num_ber), "rgb.jpg")
        x_e_path = os.path.join(self.attention_path,str(self.num_ber), "x_e.jpg")
        x_fused_path = os.path.join(self.attention_path,str(self.num_ber), "x_fused.jpg")
        os.makedirs(os.path.join(self.attention_path,str(self.num_ber)), exist_ok=True)
        cv2.imwrite(rgb_path, rgb)
        cv2.imwrite(x_e_path, x_e)
        cv2.imwrite(x_fused_path, x_fused)
        self.num_ber += 1

    @torch.no_grad()
    def feature2rgb(self, x):
        min = x.min()
        max = x.max()
        x = (x - min) / (max - min + 1e-8) * 255
        x = x.astype(np.uint8)
        x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (512, 512))
        return x


    def flops(self, x, verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanOflex": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanCore": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanNRow": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = x
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        # return sum(Gflops.values()) * 1e9
        params = params / 1e6
        flops = sum(Gflops.values())
        return f"params {params} GFLOPs {flops}"



    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_extra_path_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        kernel_size = patch_size
        stride = kernel_size // 2 + 1
        padding = kernel_size // 2

        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
            nn.GELU(),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )


    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



class Down_channel(nn.Module):
    def __init__(self, num_modals=4, dims=[96, 192, 384, 768], act_layer=nn.GELU, drop=0.0, **kwargs):
        super(Down_channel, self).__init__()
        self.ex_modals = num_modals - 1
        self.in_dim = dims[0] * self.ex_modals
        self.out_dim = dims[0]
        self.fc1 = nn.Linear(self.in_dim, self.out_dim)




    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Down_channelV2(nn.Module):
    def __init__(self, num_modals=4, dims=[96, 192, 384, 768], act_layer=nn.GELU, drop=0.0, **kwargs):
        super(Down_channelV2, self).__init__()
        self.ex_modals = num_modals - 1
        self.in_dim = dims[0] * self.ex_modals
        self.out_dim = dims[0]
        self.fc1 = nn.Linear(self.in_dim, self.out_dim)
        self.norm1 = nn.LayerNorm(self.out_dim)
        self.act = act_layer()
        self.MambaCross = MambaCross(hidden_dim=self.in_dim, ssm_ratio=2.0)
        self.vision_path = "vision_attention"
        self.number = 0


    def forward(self, x):

        # self.getcorration(x)
        #aolp = x[0]
        #dolp = x[1]
        #nir = x[2]

        x = torch.cat(x, dim=1)
        B, C, H, W = x.shape
        x = x.reshape(B, H*W, C).contiguous()
        x = self.MambaCross(x)
        x = x.reshape(B, C, H, W).contiguous()
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)

        # self.correlation_img(aolp, dolp, nir, x)
        return x


    def getcorration(self, x):
        B, C, H, W = x[0].shape
        CenH = H // 2
        CenW = W // 2

        # 存储i, corr12 和 corr13的列表
        i_values = []
        corr12_values = []
        corr13_values = []
        corr23_values = []

        x1 = x[0].reshape(B, H, W, C).contiguous()
        x2 = x[1].reshape(B, H, W, C).contiguous()
        x3 = x[2].reshape(B, H, W, C).contiguous()
        centralx1 = abs(x1[0][CenH][CenW])
        centralx2 = abs(x2[0][CenH][CenW])
        for i in range(0, H // 2):
            corrx2 = abs(x2[0][CenH+i][CenW+i])
            corrx3 = abs(x3[0][CenH+i][CenW+i])
            correlation_matrix12 = torch.corrcoef(torch.stack((centralx1, corrx2), dim=0))
            correlation_matrix13 = torch.corrcoef(torch.stack((centralx1, corrx3), dim=0))
            correlation_matrix23 = torch.corrcoef(torch.stack((corrx2, corrx3), dim=0))
            corr12 = abs(correlation_matrix12[0][1])
            corr13 = abs(correlation_matrix13[0][1])
            corr23 = abs(correlation_matrix23[0][1])
            i_values.append(i)
            corr12_values.append(corr12.item())
            corr13_values.append(corr13.item())
            corr23_values.append(corr23.item())
        # 保存为CSV
        path = "EXP/EFF_MSSM/3/" + str(self.number) + ".csv"
        with open(path, 'w') as f:
            f.write("i,corr12,corr13,corr23\n")
            for i in range(len(i_values)):
                f.write(str(i_values[i]) + "," + str(corr12_values[i]) + "," + str(corr13_values[i]) + "," + str(corr23_values[i]) + "\n")

        # plt.figure(figsize=(10, 6))
        # plt.plot(i_values, corr12_values, label='corr12', marker='o')
        # plt.plot(i_values, corr13_values, label='corr13', marker='x')
        # plt.xlabel('i')
        # plt.ylabel('Correlation Coefficient')
        # plt.title('Correlation Coefficients vs i')
        # plt.legend()
        # # 保存图像
        # path = "EXP/EFF_MSSM/1/" + str(self.number) + ".png"
        # plt.savefig(path)
        self.number += 1

    def correlation_img(self, aolp, dolp, nir, fused):
        B, C, H, W = aolp.shape
        aolp = aolp.squeeze(0).detach().cpu().numpy()
        dolp = dolp.squeeze(0).detach().cpu().numpy()
        nir = nir.squeeze(0).detach().cpu().numpy()
        fused = fused.squeeze(0).detach().cpu().numpy()
        # C, H, W to C, H*W
        aolp = aolp.reshape(C, H*W)
        dolp = dolp.reshape(C, H*W)
        nir = nir.reshape(C, H*W)
        fused = fused.reshape(C, H*W)

        fused_aolp = np.vstack((aolp, fused))
        fused_dolp = np.vstack((dolp, fused))
        fused_nir = np.vstack((nir, fused))

        corr_aolp = np.corrcoef(fused_aolp)
        corr_dolp = np.corrcoef(fused_dolp)
        corr_nir = np.corrcoef(fused_nir)

        corr_aolp = np.maximum(corr_aolp, 0)
        corr_dolp = np.maximum(corr_dolp, 0)
        corr_nir = np.maximum(corr_nir, 0)

        corr_aolp_path = os.path.join(self.vision_path, str(self.number), "aolp.png")
        corr_dolp_path = os.path.join(self.vision_path, str(self.number), "dolp.png")
        corr_nir_path = os.path.join(self.vision_path, str(self.number), "nir.png")

        os.makedirs(os.path.join(self.vision_path, str(self.number)), exist_ok=True)


        plt.imshow(corr_aolp, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.savefig(corr_aolp_path)
        plt.imshow(corr_dolp, cmap='viridis', interpolation='nearest')
        plt.savefig(corr_dolp_path)
        plt.imshow(corr_nir, cmap='viridis', interpolation='nearest')
        plt.savefig(corr_nir_path)
        plt.close()
        self.number += 1






class ConvMerge(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ConvMerge, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 1)
        self.__initweights__()

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.conv(x)
        return x

    def __initweights__(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

class CrossMambaFusionBlock(nn.Module):
    '''
    Cross Mamba Fusion (CroMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.norm = norm_layer(hidden_dim)
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = CrossMambaFusion_SS2D_SSM(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            initialize=ssm_init,
            # ==========================
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_fuse = self.op(x_rgb, x_e)
        x_fuse = x_rgb + x_e + self.drop_path1(x_fuse)
        return x_fuse


class CrossMambaFusion_SS2D_SSM(nn.Module):
    '''
    Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        k_group = 4


        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = Linear(self.d_model, self.d_inner, bias=bias)
        self.in_proj_modalx = Linear(self.d_model, self.d_inner, bias=bias)
        self.act = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        self.out_proj = Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

        self.CMA_ssm = Cross_Mamba_Attention_SSM(
            d_model=self.d_model,
            d_inner=self.d_inner,
            k_group=k_group,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            LayerNorm=LayerNorm,
            **kwargs,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        B, D, H, W = x_rgb.shape
        if self.d_conv > 1:
            x_rgb_conv = self.act(self.conv2d(x_rgb))  # (b, d, h, w)
            x_e_conv = self.act(self.conv2d(x_e))  # (b, d, h, w)
            y_rgb, y_e = self.CMA_ssm(x_rgb_conv, x_e_conv)
            x_rgb_squeeze = self.avg_pool(x_rgb_conv).contiguous().view(B, D)
            x_e_squeeze = self.avg_pool(x_e_conv).contiguous().view(B, D)
            x_rgb_exitation = self.fc1(x_rgb_squeeze).view(B, D, 1, 1).contiguous()  # b, 1, 1, d
            x_e_exitation = self.fc2(x_e_squeeze).view(B, D, 1, 1).contiguous()
            y_rgb = y_rgb * x_e_exitation
            y_e = y_e * x_rgb_exitation
            y = torch.concat([y_rgb, y_e], dim=1)
        out = self.dropout(self.out_proj(y))
        return out

class Cross_Mamba_Attention_SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_inner=192,
            k_group=4,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            LayerNorm=nn.LayerNorm,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_1_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj_1], dim=0))  # (K, N, inner)
        del self.x_proj_1

        self.x_proj_2 = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_2_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj_2], dim=0))  # (K, N, inner)
        del self.x_proj_2

        self.dt_projs_1 = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        self.dt_projs_1_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs_1], dim=0))  # (K, inner, rank)
        self.dt_projs_1_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_1], dim=0))  # (K, inner)
        del self.dt_projs_1

        self.dt_projs_2 = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        self.dt_projs_2_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs_2], dim=0))  # (K, inner, rank)
        self.dt_projs_2_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_2], dim=0))  # (K, inner)
        del self.dt_projs_2

        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner, copies=k_group, merge=True)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner, copies=k_group, merge=True)  # (D)
        self.D_1 = self.D_init(self.d_inner, copies=k_group, merge=True)  # (D)
        self.D_2 = self.D_init(self.d_inner, copies=k_group, merge=True)  # (D)

        # out norm ===================================
        self.out_norm_1 = LayerNorm(self.d_inner)
        self.out_norm_2 = LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_proj_1_weight = self.x_proj_1_weight
        x_proj_2_weight = self.x_proj_2_weight
        dt_projs_1_weight = self.dt_projs_1_weight
        dt_projs_2_weight = self.dt_projs_2_weight
        dt_projs_1_bias = self.dt_projs_1_bias
        dt_projs_2_bias = self.dt_projs_2_bias
        A_log_1 = self.A_log_1
        A_log_2 = self.A_log_2
        D_1 = self.D_1
        D_2 = self.D_2
        selective_scan = SelectiveScanOflex
        CrossScan = CrossScanTriton
        CrossMerge = CrossMergeTriton

        B, D, H, W = x_rgb.shape
        D, N = A_log_1.shape
        K, D, R = dt_projs_1_weight.shape
        L = H * W

        x_rgb = CrossScan.apply(x_rgb)
        x_e = CrossScan.apply(x_e)


        xl_rgb = F.conv1d(x_rgb.view(B, -1, L), x_proj_1_weight.view(-1, D, 1),
                         bias=None, groups=K)
        d_rgb, B_rgb, C_rgb = torch.split(xl_rgb.view(B, K, -1, L), [R, N, N], dim=2)
        d_rgb = F.conv1d(d_rgb.contiguous().view(B, -1, L), dt_projs_1_weight.view(K * D, -1, 1), groups=K)

        xl_e = F.conv1d(x_e.view(B, -1, L), x_proj_2_weight.view(-1, D, 1),
                       bias=None, groups=K)
        d_e, B_e, C_e = torch.split(xl_e.view(B, K, -1, L), [R, N, N], dim=2)
        d_e = F.conv1d(d_e.contiguous().view(B, -1, L), dt_projs_2_weight.view(K * D, -1, 1), groups=K)

        x_rgb = x_rgb.view(B, -1, L)
        d_rgb = d_rgb.contiguous().view(B, -1, L)
        A_rgb = -torch.exp(A_log_1.float())
        B_rgb = B_rgb.contiguous().view(B, K, N, L)
        C_rgb = C_rgb.contiguous().view(B, K, N, L)
        D_rgb = D_1.to(torch.float) # (K * c)
        delta_bias_rgb = dt_projs_1_bias.view(-1).to(torch.float)

        x_e = x_e.view(B, -1, L)
        d_e = d_e.contiguous().view(B, -1, L)
        A_e = -torch.exp(A_log_2.float())
        B_e = B_e.contiguous().view(B, K, N, L)
        C_e = C_e.contiguous().view(B, K, N, L)
        D_e = D_2.to(torch.float)
        delta_bias_e = dt_projs_2_bias.view(-1).to(torch.float)

        y_rgb = selective_scan.apply(
            x_rgb, d_rgb,
            A_rgb, B_rgb, C_e, D_e,
            delta_bias_rgb,
            True,
            -1,-1,True
        )
        y_rgb = y_rgb.view(B, K, -1, H, W)
        y_e = selective_scan.apply(
            x_e, d_e,
            A_e, B_e, C_rgb, D_rgb,
            delta_bias_e,
            True,
            -1, -1, True
        )
        y_e = y_e.view(B, K, -1, H, W)

        y_rgb = CrossMerge.apply(y_rgb).view(B, -1, H, W)
        y_e = CrossMerge.apply(y_e).view(B, -1, H, W)
        y_rgb = self.out_norm_1(y_rgb)
        y_e = self.out_norm_2(y_e)
        return y_rgb, y_e

class down_SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_inner=192,
            k_group=4,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            LayerNorm=nn.LayerNorm,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_1_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj_1], dim=0))  # (K, N, inner)
        del self.x_proj_1


        self.dt_projs_1 = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        self.dt_projs_1_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs_1], dim=0))  # (K, inner, rank)
        self.dt_projs_1_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_1], dim=0))  # (K, inner)
        del self.dt_projs_1


        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner, copies=k_group, merge=True)  # (D, N)
        self.D_1 = self.D_init(self.d_inner, copies=k_group, merge=True)  # (D)

        # out norm ===================================
        self.out_norm_1 = LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_proj_1_weight = self.x_proj_1_weight
        dt_projs_1_weight = self.dt_projs_1_weight
        dt_projs_1_bias = self.dt_projs_1_bias
        A_log_1 = self.A_log_1
        D_1 = self.D_1
        selective_scan = SelectiveScanOflex
        CrossScan = CrossScanTriton

        B, D, H, W = x_rgb.shape
        D, N = A_log_1.shape
        K, D, R = dt_projs_1_weight.shape
        L = H * W

        x_rgb = CrossScan.apply(x_rgb)


        xl_rgb = F.conv1d(x_rgb.view(B, -1, L), x_proj_1_weight.view(-1, D, 1),
                         bias=None, groups=K)
        d_rgb, B_rgb, C_rgb = torch.split(xl_rgb.view(B, K, -1, L), [R, N, N], dim=2)
        d_rgb = F.conv1d(d_rgb.contiguous().view(B, -1, L), dt_projs_1_weight.view(K * D, -1, 1), groups=K)


        x_rgb = x_rgb.view(B, -1, L)
        d_rgb = d_rgb.contiguous().view(B, -1, L)
        A_rgb = -torch.exp(A_log_1.float())
        B_rgb = B_rgb.contiguous().view(B, K, N, L)
        C_rgb = C_rgb.contiguous().view(B, K, N, L)
        D_rgb = D_1.to(torch.float) # (K * c)
        delta_bias_rgb = dt_projs_1_bias.view(-1).to(torch.float)


        y_rgb = selective_scan.apply(
            x_rgb, d_rgb,
            A_rgb, B_rgb, C_rgb, D_rgb,
            delta_bias_rgb,
            True,
            -1,-1,True
        )
        y_rgb = y_rgb.view(B, K, -1, H, W)
        y_rgb = torch.cat(y_rgb, dim=1)

        return y_rgb

if __name__ == '__main__':
    x = [torch.zeros(1, 3, 512, 512), torch.ones(1, 3, 512, 512), torch.ones(1, 3, 512, 512), torch.ones(1, 3, 512, 512)]
    for i in range(len(x)):
        x[i] = x[i].cuda()
    model = MulMamba(modals = ['rgb', 'A', 'D', 'N'])
    model.cuda()
    print(model.flops(x))




