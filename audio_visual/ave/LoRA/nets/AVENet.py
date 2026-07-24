##################################
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
from einops import rearrange, repeat
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
import vision_backbone
from SSLAM import sslam_backbone
"""
We used the timm folder(no timm installation required) for network and imagenet pretrained weight
https://github.com/huggingface/pytorch-image-models/tree/main/timm

"""
class AVENet(nn.Module):

    def __init__(self, opt):
        super(AVENet, self).__init__()

        self.opt = opt

        if opt.vis_encoder_type == "vit_b":
            self.vit_v =  vision_backbone.get_timm_vit_b_pretrained_model(n_classes=1000, imgnet=True, use_lora=opt.use_lora, lora_r=opt.lora_r, lora_alpha=opt.lora_alpha)
        elif opt.vis_encoder_type == "dino_b":
            self.vit_v =  vision_backbone.get_timm_dino_b_pretrained_model(n_classes=1000, imgnet=True, use_lora=opt.use_lora, lora_r=opt.lora_r, lora_alpha=opt.lora_alpha)
        elif opt.vis_encoder_type == "dino_l":
            self.vit_v =  vision_backbone.get_timm_dino_l_pretrained_model(n_classes=1000, imgnet=True, use_lora=opt.use_lora, lora_r=opt.lora_r, lora_alpha=opt.lora_alpha)
        elif opt.vis_encoder_type == "eva_l":
            self.vit_v =  vision_backbone.get_timm_EVA_pretrained_model(n_classes=1000, imgnet=True, use_lora=opt.use_lora, lora_r=opt.lora_r, lora_alpha=opt.lora_alpha)
        else:
            raise ValueError(f"Unknown vis_encoder_type {opt.vis_encoder_type}")
        
    

        self.sslam_a = sslam_backbone.get_sslam_backbone(use_lora=True, lora_r=opt.lora_r, lora_alpha=opt.lora_alpha)

        if opt.use_lora:
            print("Using LoRA")
            import loralib as lora
            lora.mark_only_lora_as_trainable(self.vit_v, bias='none') # set bias to none to only train lora B & A
            lora.mark_only_lora_as_trainable(self.sslam_a, bias='none')	
    
        if  opt.vis_encoder_type == "vit_b" or opt.vis_encoder_type == "dino_b":
            self.mlp_class = nn.Linear(768+768, 512)
        elif opt.vis_encoder_type == "dino_l" or opt.vis_encoder_type == "eva_l":
            self.mlp_class = nn.Linear(1024+768, 512) 
   
        self.mlp_class_2 = nn.Linear(512, 29)


    def foward_normal(self, audio, vis):



      
        # vision
        vis = rearrange(vis, 'b t c w h -> (b t) c w h')
        f_v = self.vit_v.patch_embed(vis)
        f_v = self.vit_v._pos_embed(f_v)
        f_v = self.vit_v.patch_drop(f_v)
        f_v = self.vit_v.norm_pre(f_v)

        # Audio

        audio = repeat(audio, 'b t len dim -> b t c len dim', c=1) 
        audio = rearrange(audio, 'b t c w h -> (b t) c w h')
        
 
    
        afeat_extractor = self.sslam_a.model.modality_encoders["IMAGE"]
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

   

        if self.sslam_a.model.dropout_input is not None:
            f_a =  self.sslam_a.model.dropout_input(f_a)

        if len(self.vit_v.blocks) != len(self.sslam_a.model.blocks):
            if (len(self.vit_v.blocks)/len(self.sslam_a.model.blocks)) != 2:
                raise ValueError(f"len(self.vit_v.blocks) {len(self.vit_v.blocks)} != len(self.sslam_a.model.blocks) {len(self.sslam_a.model.blocks)}")
        

            audio_blks = [self.sslam_a.model.blocks[i//2] if i%2 !=0 else None for i in range(len(self.vit_v.blocks)) ]
            for idx_stage, (vit_blk,sslam_blk) in enumerate(zip(self.vit_v.blocks,audio_blks)):
                if sslam_blk is not None:

                    #Vision
                    # Attention
                    f_v_shortcut1 = f_v.clone()
                    f_v = vit_blk.norm1(f_v)
                    f_v = vis_qkv_attn(vit_blk.attn, f_v)
                    f_v = vis_o_attn(vit_blk.attn, f_v)
                    f_v = f_v_shortcut1 + vit_blk.drop_path1(vit_blk.ls1(f_v))
                    # MLP
                    f_v_shortcut2 = f_v.clone()
                    f_v = vit_blk.norm2(f_v)
                    f_v = vis_mlp(vit_blk.mlp, f_v)
                    f_v = f_v_shortcut2 + vit_blk.drop_path2(vit_blk.ls2(f_v))


                    #Audio
                    # Attention
                    f_a_shortcut1 = f_a.clone()
                    f_a = audio_qkv_attn(sslam_blk.attn, f_a,
                                        padding_mask=f_a_masked_padding_mask,
                                        alibi_bias=f_a_alibi_scale)
                    f_a = audio_o_attn(sslam_blk.attn, f_a)
                    f_a = sslam_blk.norm1(f_a_shortcut1 + sslam_blk.drop_path(f_a))
                    # MLP
                    f_a_shortcut2 = f_a.clone()
                    f_a = audio_mlp(sslam_blk.mlp, f_a)
                    f_a = sslam_blk.norm2(f_a_shortcut2 + sslam_blk.drop_path(sslam_blk.post_mlp_dropout(f_a)))

                else:
                    #Vision
                    # Attention
                    f_v_shortcut1 = f_v.clone()
                    f_v = vit_blk.norm1(f_v)
                    f_v = vis_qkv_attn(vit_blk.attn, f_v)
                    f_v = vis_o_attn(vit_blk.attn, f_v)
                    f_v = f_v_shortcut1 + vit_blk.drop_path1(vit_blk.ls1(f_v))
                    # MLP
                    f_v_shortcut2 = f_v.clone()
                    f_v = vit_blk.norm2(f_v)
                    f_v = vis_mlp(vit_blk.mlp, f_v)
                    f_v = f_v_shortcut2 + vit_blk.drop_path2(vit_blk.ls2(f_v))

        else:
            for i ,  (vit_blk,sslam_blk) in enumerate(zip(self.vit_v.blocks,self.sslam_a.model.blocks)):
                

                #Vision
                # Attention
                f_v_shortcut = f_v.clone()
                f_v = vit_blk.norm1(f_v)
                f_v = vis_qkv_attn(vit_blk.attn, f_v)
                f_v = vis_o_attn(vit_blk.attn, f_v)
                f_v = f_v_shortcut + vit_blk.drop_path1(vit_blk.ls1(f_v))
                # MLP
                f_v_shortcut2 = f_v.clone()
                f_v = vit_blk.norm2(f_v)
                f_v = vis_mlp(vit_blk.mlp, f_v)
                f_v = f_v_shortcut2 + vit_blk.drop_path2(vit_blk.ls2(f_v))


                #Audio
                # Attention
                f_a_shortcut = f_a.clone()
                f_a = audio_qkv_attn(sslam_blk.attn, f_a,
                                     padding_mask=f_a_masked_padding_mask,
                                     alibi_bias=f_a_alibi_scale)
                f_a = audio_o_attn(sslam_blk.attn, f_a)
                f_a = sslam_blk.norm1(f_a_shortcut + sslam_blk.drop_path(f_a))
                # MLP
                f_a_shortcut2 = f_a.clone()
                f_a = audio_mlp(sslam_blk.mlp, f_a)
                f_a = sslam_blk.norm2(f_a_shortcut2 + sslam_blk.drop_path(sslam_blk.post_mlp_dropout(f_a)))

        
        f_v = self.vit_v.norm(f_v) 
        f_v = f_v[:, 0] # get cls

        if self.sslam_a.model.norm is not None:
            f_a = self.self.sslam_a.model.norm(f_a)
        f_a = f_a[:,: afeat_extractor.modality_cfg.num_extra_tokens] # (B,1,dim)
        f_a = f_a.squeeze(1)  # (B,dim)
        



        out_av = torch.cat((f_v, f_a), dim=-1)
        p_av = self.mlp_class(out_av)
        p_av = self.mlp_class_2(p_av)
        # due to BCEWithLogitsLoss
        p_av = F.softmax(p_av, dim=-1)
        return p_av 


    def forward(self, audio, vis):
        return self.foward_normal(audio, vis)

def vis_qkv_attn(attn_module,x):
    B, N, C = x.shape
    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn_module.q_norm(q), attn_module.k_norm(k)

    if attn_module.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=attn_module.attn_drop.p,
        )
    else:
        q = q * attn_module.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = attn_module.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    return x

def vis_o_attn(attn_module, x):

    x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x

def vis_mlp(mlp_module, x):
    x = mlp_module.fc1(x)
    x = mlp_module.act(x)
    x = mlp_module.drop1(x)
    x = mlp_module.norm(x)
    x = mlp_module.fc2(x)
    x = mlp_module.drop2(x)
    return x

def audio_qkv_attn(attn_module, x, padding_mask=None, alibi_bias=None):
    B, N, C = x.shape
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

def audio_o_attn(attn_module, x):
    x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x

def audio_mlp(mlp_module, x):
    x = mlp_module.fc1(x)
    x = mlp_module.act(x)
    x = mlp_module.drop1(x)
    x = mlp_module.norm(x)
    x = mlp_module.fc2(x)
    x = mlp_module.drop2(x)
    return x