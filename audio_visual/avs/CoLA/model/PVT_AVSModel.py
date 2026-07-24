import torch
import torch.nn as nn
import torchvision.models as models
# from model.pvt import pvt_v2_b5
from model.TPAVI import TPAVIModule
from ipdb import set_trace
import timm
import torch.nn.functional as F
from einops import rearrange, repeat
import model_backbone
import math
from typing import Callable, Optional, Tuple, Union, List
from SSLAM import sslam_backbone



def audio_qkv_attn(attn_module, x, padding_mask=None, alibi_bias=None, x_c=None):
    B, N, C = x.shape
    if x_c is not None:

        qkv = (
            attn_module.qkv(x, x_c=x_c, is_sequence=True)
            .reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
            .permute(2, 0, 3, 1, 4)  # qkv x B x H x L x D
        )	
    else:
        qkv = (
            attn_module.qkv(x)
            .reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
            .permute(2, 0, 3, 1, 4)  # qkv x B x H x L x D
        )
    q, k, v = (
        qkv[0],
        qkv[1],
        qkv[2],
    )  # make torchscript happy (cannot use tensor as tuple)

    dtype = q.dtype

    q = q * attn_module.scale
    attn = q @ k.transpose(-2, -1)

    if alibi_bias is not None:
        attn = attn.type_as(alibi_bias)
        attn[:, : alibi_bias.size(1)] += alibi_bias

    if padding_mask is not None and padding_mask.any():
        attn = attn.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float("-inf"),
        )

    attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
    attn = attn_module.attn_drop(attn)
    x = (attn @ v).transpose(1, 2)  #
    x = x.reshape(B, N, C)

    return x

def audio_o_attn(attn_module, x, x_c=None):
    if x_c is not None:
   
        x = attn_module.proj(x, x_c=x_c, is_sequence=True)
    else:
        x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x

def audio_mlp(mlp_module, x, x_c=None):
    if x_c is not None:
        x = mlp_module.fc1(x, x_c=x_c, is_sequence=True)
    else:
        x = mlp_module.fc1(x)
    x = mlp_module.act(x)
    x = mlp_module.drop1(x)
    x = mlp_module.norm(x)
    if x_c is not None:
        x = mlp_module.fc2(x, x_c=x_c, is_sequence=True)
    else:
        x = mlp_module.fc2(x)
    x = mlp_module.drop2(x)
    return x

def vis_window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def vis_window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x

def vis_partition(block_module, x):
    B, H, W, C = x.shape
    # cyclic shift
    has_shift = any(block_module.shift_size)
    if has_shift:
        shifted_x = torch.roll(x, shifts=(-block_module.shift_size[0], -block_module.shift_size[1]), dims=(1, 2))
    else:
        shifted_x = x
    x_windows = vis_window_partition(shifted_x, block_module.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, block_module.window_area, C)  # nW*B, window_size*window_size, C

    return x_windows

def vis_unpartition(block_module, attn_windows, vC):
    has_shift = any(block_module.shift_size)
    # merge windows
    attn_windows = attn_windows.view(-1, block_module.window_size[0], block_module.window_size[1], vC)
    shifted_x = vis_window_reverse(attn_windows, block_module.window_size, block_module.input_resolution)  # B H' W' C

    # reverse cyclic shift
    if has_shift:
        x = torch.roll(shifted_x, shifts=block_module.shift_size, dims=(1, 2))
    else:
        x = shifted_x
    return x
    
def vis_qkv_attn(attn_module, x, mask=None, x_c =None):

    B_, N, C = x.shape
    qkv_bias = None
    if attn_module.q_bias is not None:
        qkv_bias = torch.cat((attn_module.q_bias, attn_module.k_bias, attn_module.v_bias))
    
    if x_c is not None:
        qkv =  attn_module.qkv(x, x_c=x_c,partition_feat=True , is_sequence=True) + qkv_bias
    else:
        qkv =  attn_module.qkv(x) + qkv_bias
    qkv = qkv.reshape(B_, N, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # cosine attention
    attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
    logit_scale = torch.clamp(attn_module.logit_scale, max=math.log(1. / 0.01)).exp()
    attn = attn * logit_scale

    relative_position_bias_table = attn_module.cpb_mlp(attn_module.relative_coords_table).view(-1, attn_module.num_heads)
    relative_position_bias = relative_position_bias_table[attn_module.relative_position_index.view(-1)].view(
        attn_module.window_size[0] * attn_module.window_size[1], attn_module.window_size[0] * attn_module.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        num_win = mask.shape[0]
        attn = attn.view(-1, num_win, attn_module.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, attn_module.num_heads, N, N)
        attn = attn_module.softmax(attn)
    else:
        attn = attn_module.softmax(attn)

    attn = attn_module.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

    return x

def vis_o_attn(attn_module, x, x_c=None):
    # output projection

    if x_c is not None:
        x = attn_module.proj(x, x_c=x_c, partition_feat=True,is_sequence=True)
    else:
        x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x

def vis_mlp(mlp_module, x, x_c=None):

    # MLP
    if x_c is not None:
        x = mlp_module.fc1(x, x_c=x_c, is_sequence=True)
    else:
        x = mlp_module.fc1(x)
    x = mlp_module.act(x)
    x = mlp_module.drop1(x)
    x = mlp_module.norm(x)
    if x_c is not None:
        x = mlp_module.fc2(x, x_c=x_c, is_sequence=True)
    else:
        x = mlp_module.fc2(x)
    x = mlp_module.drop2(x)
    return x

class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x

class Pred_endecoder(nn.Module):
    # pvt-v2 based encoder decoder
    def __init__(self, channel=256,opt=None, config=None, vis_dim=[64, 128, 320, 512], tpavi_stages=[], tpavi_vv_flag=False, tpavi_va_flag=True):
        super(Pred_endecoder, self).__init__()
        self.cfg = config
        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag
        self.vis_dim = vis_dim
        self.opt = opt
        
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[3])
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[2])
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[1])
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[0])

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)

        self.x1_linear = nn.Linear(192,64)
        self.x2_linear = nn.Linear(384,128)
        self.x3_linear = nn.Linear(768,320)
        self.x4_linear = nn.Linear(1536,512)

        self.audio_linear = nn.Linear(768,128)


        lora_fusion_swin = {
            0 :[False,True], # 192
            1 :[False,True], #384
            2 :[False, True,  #768
                False, True,  
                False, True,
                False, True,  
                False, True, 
                False, True,  
                False, True,
                False, True,
                False, True,  #,1536
                ],
            3 :[False,True], #,1536
        }

        sslam_dim = (768,768,768, 768)
        self.swin_v = model_backbone.get_timm_swinv2_pretrained_model(n_classes=1000, imgnet=True, use_lora=opt.use_lora, lora_r=opt.lora_r, lora_alpha=opt.lora_alpha,  lora_c_dim=sslam_dim ,lora_fusion_layers = lora_fusion_swin, lora_reduction=opt.lora_reduction, lora_c_scaling=opt.lora_c_scaling)
        
        
        lora_fusion_aud= [True] *12

        vis_dim=   [192,384] + ([768]*9) + [1536] #(192,384,768,1536)

        sslam_redcution = [int((x/192)*opt.lora_reduction) for x in vis_dim]

        self.audio_a = sslam_backbone.get_sslam_backbone(
                                                        use_lora=opt.use_lora, 
                                                        lora_r=opt.lora_r,
                                                        lora_alpha=opt.lora_alpha, 
                                                        lora_c_dim=vis_dim,
                                                        lora_fusion_layers=lora_fusion_aud, 
                                                        lora_reduction=sslam_redcution,
                                                        lora_c_scaling=opt.lora_c_scaling
                                                        )

        if opt.use_lora:
            print("Using LoRA")
            import loralib as lora
            lora.mark_only_lora_as_trainable(self.swin_v, bias='none') # set bias to none to only train lora B & A
            lora.mark_only_lora_as_trainable(self.audio_a, bias='none')		

        # print("Swin")
        # for n,p in self.swin_v.named_parameters():
        # 	if p.requires_grad:
        # 		print(f"Named : {n} , paramters {p.shape}")
        # print("loaded SSLAM")
        # for n,p in self.audio_a.named_parameters():
        # 	if p.requires_grad:
        # 		print(f"Named : {n} , paramters {p.shape}")
        # exit()

        for i in self.tpavi_stages:
            setattr(self, f"tpavi_b{i+1}", TPAVIModule(in_channels=channel, mode='dot'))
            print("==> Build TPAVI block...")

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )


    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        try:
            x = x.reshape(-1, 5, C, H, W)
        except:
            print("pre_reshape_for_tpavi: ", x.shape)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def tpavi_vv(self, x, stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, _ = tpavi_b(x) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x

    def tpavi_va(self, x, audio, stage):
        # x: visual, [B*T, C=256, H, W]
        # audio: [B*T, 128]
        # ra_flag: return audio feature list or not
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        try:
            audio = audio.view(-1, 5, audio.shape[-1]) # [B, T, 128]
        except:
            print("tpavi_va: ", audio.shape)
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, a = tpavi_b(x, audio) # [B, C, T, H, W], [B, T, C]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x, a

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, audio_feature=None):
        # ---------> yb:add
        B, frame, C, H, W = x.shape
        x = x.view(B*frame, C, H, W)
        x = F.interpolate(x, mode='bicubic',size=[192,192])

        # # for debugging fowrard
        # self.swin_v.eval()
        # self.audio_a.eval()

        audio = repeat(audio_feature, 'b t len dim -> b t c len dim', c=1) 
        audio = rearrange(audio, 'b t c w h -> (b t) c w h')
        
        # ffv = self.swin_v(x)
        # ffa = self.audio_a(audio)


        f_v = self.swin_v.patch_embed(x)
        # # For debugging forward of pred_endecoder
        
        afeat_extractor = self.audio_a.model.modality_encoders["IMAGE"]
        a_extractor_out = afeat_extractor(
            audio, 
            None, # padding_mask
            False, # mask
            remove_masked=False,
            clone_batch=1,
            mask_seeds=None,
            precomputed_mask=None
            )
        f_a = a_extractor_out["x"]
        f_a_encoder_mask = a_extractor_out["encoder_mask"]
        f_a_masked_padding_mask = a_extractor_out["padding_mask"]
        f_a_masked_alibi_bias = a_extractor_out.get("alibi_bias", None)
        f_a_alibi_scale = a_extractor_out.get("alibi_scale", None)
        if self.audio_a.model.dropout_input is not None:
            f_a =  self.audio_a.model.dropout_input(f_a)
        
        num_a_blocks = len(self.audio_a.model.blocks)
        num_s_blocks = 0
        for sl in self.swin_v.layers:
            num_s_blocks+=len(sl.blocks)
        

        # exit()
        blk_a = [self.audio_a.model.blocks[i//2] if i%2 !=0 else None for i in range(num_s_blocks) ]	
        multi_scale = []
        blocks_count = 0
        for idx_stage, swin_vl in enumerate(self.swin_v.layers):
            # in layer
            f_v = swin_vl.downsample(f_v)
            swin_blocks = swin_vl.blocks

            for  blk_v in swin_blocks:
        
                if 	blk_a[blocks_count] is not None:


                    # Vision Part
                    vB, vH, vW, vC = f_v.shape
                    # audio 
                    sslam_blk = blk_a[blocks_count]
                    # Attention 
                    f_v_shortcut1 = f_v.clone()
                    f_a_shortcut1 = f_a.clone()
                    ### QKV attention Fusion
                    fusion_vis =  f_v.reshape(f_v.shape[0],-1, f_v.shape[-1]).mean(dim=1)
                    fusion_aud = f_a[:,: afeat_extractor.modality_cfg.num_extra_tokens].squeeze(1) 
        
                    # vision
                    f_v = vis_partition(blk_v, f_v)
                    f_v = vis_qkv_attn(blk_v.attn, f_v, mask=blk_v.attn_mask, x_c=fusion_aud)
                    # audio 
                    f_a = audio_qkv_attn(sslam_blk.attn, f_a,
                                        padding_mask=f_a_masked_padding_mask,
                                        alibi_bias=f_a_alibi_scale,
                                        x_c = fusion_vis
                                        )
                    
                    ### Output projection fusion

                    fusion_vis2 = vis_unpartition(blk_v, f_v ,vC=vC)
                    fusion_vis2 =  fusion_vis2.reshape(fusion_vis2.shape[0],-1, fusion_vis2.shape[-1]).mean(dim=1)
                    fusion_aud2 = f_a[:,: afeat_extractor.modality_cfg.num_extra_tokens].squeeze(1) 

                    # Vision
                    f_v = vis_o_attn(blk_v.attn, f_v, x_c=fusion_aud2 )
                    f_v = vis_unpartition(blk_v, f_v ,vC=vC)
                    f_v = f_v_shortcut1 + blk_v.drop_path1(blk_v.norm1(f_v))
                    f_v = f_v.reshape(vB, -1, vC)  # reshape back to original shape
                    # Audio
                    f_a = audio_o_attn(sslam_blk.attn, f_a, x_c=fusion_vis2)
                    f_a = sslam_blk.norm1(f_a_shortcut1 + sslam_blk.drop_path(f_a))

                    # FFN
                    f_v_shortcut2 = f_v.clone()
                    f_a_shortcut2 = f_a.clone()
                    # FFN fusion
                    fusion_vis3 = f_v.mean(dim=1)
                    fusion_aud3 = f_a[:,: afeat_extractor.modality_cfg.num_extra_tokens].squeeze(1) 
      
                    # Vision
                    f_v = vis_mlp(blk_v.mlp, f_v, x_c=fusion_aud3)
                    f_v = f_v_shortcut2 + blk_v.drop_path2(blk_v.norm2(f_v)) 
                    f_v = f_v.reshape(vB, vH, vW, vC)  # reshape back to original shape	
                    # Audio
                    f_a_shortcut2 = f_a.clone()
                    f_a = audio_mlp(sslam_blk.mlp, f_a, x_c=fusion_vis3)
                    f_a = sslam_blk.norm2(f_a_shortcut2 + sslam_blk.drop_path(sslam_blk.post_mlp_dropout(f_a)))

        
                else:

                    vB, vH, vW, vC = f_v.shape

                    # Attention 
                    f_v_shortcut1 = f_v.clone()
                    ### QKV attention
                    f_v = vis_partition(blk_v, f_v)
                    f_v = vis_qkv_attn(blk_v.attn, f_v, mask=blk_v.attn_mask)
                    ### Output projection
                    f_v = vis_o_attn(blk_v.attn, f_v)
                    f_v = vis_unpartition(blk_v, f_v ,vC=vC)
                    f_v = f_v_shortcut1 + blk_v.drop_path1(blk_v.norm1(f_v))
                    f_v = f_v.reshape(vB, -1, vC)  # reshape back to original shape
                        
                    # FFN
                    f_v_shortcut2 = f_v.clone()
                    f_v = vis_mlp(blk_v.mlp, f_v)
                    f_v = f_v_shortcut2 + blk_v.drop_path2(blk_v.norm2(f_v))
                    f_v = f_v.reshape(vB, vH, vW, vC)  # reshape back to original shape	
                
                # increaste the blocj count of swin
                blocks_count+=1



            if idx_stage != 3:
                multi_scale.append(f_v)
            else:
                multi_scale.append(self.swin_v.norm(f_v))
    
        f_v = self.swin_v.norm(f_v)

        if self.audio_a.model.norm is not None:
            f_a = self.self.audio_a.model.norm(f_a)

        f_a = f_a[:,: afeat_extractor.modality_cfg.num_extra_tokens] # (B,1,dim)
        f_a = f_a.squeeze(1)  # (B,dim)
        

        audio_feature = rearrange(f_a, '(b t) d -> b t d', t=5)
        audio_feature = self.audio_linear(audio_feature)

 
        x1 = multi_scale[0]  
        x2 = multi_scale[1]  
        x3 = multi_scale[2]  
        x4 = multi_scale[3] 


        x1 = self.x1_linear(x1)
        x2 = self.x2_linear(x2)
        x3 = self.x3_linear(x3)
        x4 = self.x4_linear(x4)


        x1 = F.interpolate(rearrange(x1, 'BF w h c -> BF c w h'), mode='bicubic',size=[56,56])
        x2 = F.interpolate(rearrange(x2, 'BF w h c -> BF c w h'), mode='bicubic',size=[28,28])
        x3 = F.interpolate(rearrange(x3, 'BF w h c -> BF c w h'), mode='bicubic',size=[14,14])
        x4 = F.interpolate(rearrange(x4, 'BF w h c -> BF c w h'), mode='bicubic',size=[7,7])

        conv1_feat = self.conv1(x1)    # BF x 256 x 56 x 56
        conv2_feat = self.conv2(x2)    # BF x 256 x 28 x 28
        conv3_feat = self.conv3(x3)    # BF x 256 x 14 x 14
        conv4_feat = self.conv4(x4)    # BF x 256 x  7 x  7

        feature_map_list = [conv1_feat, conv2_feat, conv3_feat, conv4_feat]
        a_fea_list = [None] * 4

        if len(self.tpavi_stages) > 0:
            if (not self.tpavi_vv_flag) and (not self.tpavi_va_flag):
                raise Exception('tpavi_vv_flag and tpavi_va_flag cannot be False at the same time if len(tpavi_stages)>0, \
                    tpavi_vv_flag is for video self-attention while tpavi_va_flag indicates the standard version (audio-visual attention)')
            for i in self.tpavi_stages:
                tpavi_count = 0
                conv_feat = torch.zeros_like(feature_map_list[i]).cuda()
                if self.tpavi_vv_flag:
                    conv_feat_vv = self.tpavi_vv(feature_map_list[i], stage=i)
                    conv_feat += conv_feat_vv
                    tpavi_count += 1
                if self.tpavi_va_flag:
                    conv_feat_va, a_fea = self.tpavi_va(feature_map_list[i], audio_feature, stage=i)
                    conv_feat += conv_feat_va
                    tpavi_count += 1
                    a_fea_list[i] = a_fea
                conv_feat /= tpavi_count
                feature_map_list[i] = conv_feat # update features of stage-i which conduct non-local

        conv4_feat = self.path4(feature_map_list[3])            # BF x 256 x 14 x 14
        conv43 = self.path3(conv4_feat, feature_map_list[2])    # BF x 256 x 28 x 28
        conv432 = self.path2(conv43, feature_map_list[1])       # BF x 256 x 56 x 56
        conv4321 = self.path1(conv432, feature_map_list[0])     # BF x 256 x 112 x 112

        pred = self.output_conv(conv4321)   # BF x 1 x 224 x 224
        # print(pred.shape)

        return pred, feature_map_list, a_fea_list


if __name__ == "__main__":
    imgs = torch.randn(10, 3, 224, 224)
    audio = torch.randn(2, 5, 128)
    # model = Pred_endecoder(channel=256)
    model = Pred_endecoder(channel=256, tpavi_stages=[0,1,2,3], tpavi_va_flag=True,)
    # output = model(imgs)
    output = model(imgs, audio)
    