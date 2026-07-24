# NOTE custom_timm.models.layers is DEPRECATED, please use custom_timm.layers, this is here to reduce breakages in transition
from custom_timm.layers.activations import *
from custom_timm.layers.adaptive_avgmax_pool import \
    adaptive_avgmax_pool2d, select_adaptive_pool2d, AdaptiveAvgMaxPool2d, SelectAdaptivePool2d
from custom_timm.layers.attention_pool2d import AttentionPool2d, RotAttentionPool2d, RotaryEmbedding
from custom_timm.layers.blur_pool import BlurPool2d
from custom_timm.layers.classifier import ClassifierHead, create_classifier
from custom_timm.layers.cond_conv2d import CondConv2d, get_condconv_initializer
from custom_timm.layers.config import is_exportable, is_scriptable, is_no_jit, set_exportable, set_scriptable, set_no_jit,\
    set_layer_config
from custom_timm.layers.conv2d_same import Conv2dSame, conv2d_same
from custom_timm.layers.conv_bn_act import ConvNormAct, ConvNormActAa, ConvBnAct
from custom_timm.layers.create_act import create_act_layer, get_act_layer, get_act_fn
from custom_timm.layers.create_attn import get_attn, create_attn
from custom_timm.layers.create_conv2d import create_conv2d
from custom_timm.layers.create_norm import get_norm_layer, create_norm_layer
from custom_timm.layers.create_norm_act import get_norm_act_layer, create_norm_act_layer, get_norm_act_layer
from custom_timm.layers.drop import DropBlock2d, DropPath, drop_block_2d, drop_path
from custom_timm.layers.eca import EcaModule, CecaModule, EfficientChannelAttn, CircularEfficientChannelAttn
from custom_timm.layers.evo_norm import EvoNorm2dB0, EvoNorm2dB1, EvoNorm2dB2,\
    EvoNorm2dS0, EvoNorm2dS0a, EvoNorm2dS1, EvoNorm2dS1a, EvoNorm2dS2, EvoNorm2dS2a
from custom_timm.layers.fast_norm import is_fast_norm, set_fast_norm, fast_group_norm, fast_layer_norm
from custom_timm.layers.filter_response_norm import FilterResponseNormTlu2d, FilterResponseNormAct2d
from custom_timm.layers.gather_excite import GatherExcite
from custom_timm.layers.global_context import GlobalContext
from custom_timm.layers.helpers import to_ntuple, to_2tuple, to_3tuple, to_4tuple, make_divisible, extend_tuple
from custom_timm.layers.inplace_abn import InplaceAbn
from custom_timm.layers.linear import Linear
from custom_timm.layers.mixed_conv2d import MixedConv2d
from custom_timm.layers.mlp import Mlp, GluMlp, GatedMlp, ConvMlp
from custom_timm.layers.non_local_attn import NonLocalAttn, BatNonLocalAttn
from custom_timm.layers.norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d
from custom_timm.layers.norm_act import BatchNormAct2d, GroupNormAct, convert_sync_batchnorm
from custom_timm.layers.padding import get_padding, get_same_padding, pad_same
from custom_timm.layers.patch_embed import PatchEmbed
from custom_timm.layers.pool2d_same import AvgPool2dSame, create_pool2d
from custom_timm.layers.squeeze_excite import SEModule, SqueezeExcite, EffectiveSEModule, EffectiveSqueezeExcite
from custom_timm.layers.selective_kernel import SelectiveKernel
from custom_timm.layers.separable_conv import SeparableConv2d, SeparableConvNormAct
from custom_timm.layers.space_to_depth import SpaceToDepthModule
from custom_timm.layers.split_attn import SplitAttn
from custom_timm.layers.split_batchnorm import SplitBatchNorm2d, convert_splitbn_model
from custom_timm.layers.std_conv import StdConv2d, StdConv2dSame, ScaledStdConv2d, ScaledStdConv2dSame
from custom_timm.layers.test_time_pool import TestTimePoolHead, apply_test_time_pool
from custom_timm.layers.trace_utils import _assert, _float_to_int
from custom_timm.layers.weight_init import trunc_normal_, trunc_normal_tf_, variance_scaling_, lecun_normal_

import warnings
warnings.warn(f"Importing from {__name__} is deprecated, please import via custom_timm.layers", DeprecationWarning)
