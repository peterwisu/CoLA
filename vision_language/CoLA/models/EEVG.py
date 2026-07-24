import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_model.visual_backbone import build_visual
from .language_model.bert import build_bert
from torch import Tensor
from typing import Optional
import math
from typing import Dict, List
from utils.misc import nested_tensor_from_tensor_list

PATCH_LEN, ELIMINATED_THRESHOLD = None, 0.015
PATCH_LEN_DICT = {
    "SwinT": 32,
    "SwinT-S": 32,
    "ViTDet": 28,
    "DarkNet53": 20,
    "ResNet101": 20
}



class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class EEVG(nn.Module):
    def __init__(self, args):
        super(EEVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        self.num_visu_token = 32 ** 2 if "SwinT" in args.backbone else 28 ** 2
        self.num_text_token = args.max_query_len
        global PATCH_LEN
        PATCH_LEN = PATCH_LEN_DICT[args.backbone]
        # set the CoLA fusion to all layer of vision and text backbone
        args.vision_fusion  = [True] *12
        args.text_fusion = [True] * 12
        args.vislora_c_feats = 768
        args.textlora_c_feats = 768
        self.visumodel = build_visual(args)
        self.textmodel = build_bert(args)
        # for p, n in self.visumodel.named_parameters():
        #     print(p, n.shape)
        # for p, n in self.textmodel.named_parameters():
        #     print(p, n.shape)
        self.enc_num = args.bert_enc_num
        if args.use_lora:
            print("Using LoRA")

            import loralib as lora
            lora.mark_only_lora_as_trainable(self.visumodel, bias='none') # set bias to none to only train lora B & A
            lora.mark_only_lora_as_trainable(self.textmodel, bias='none')
            

        self.is_segment = args.is_segment

        print('is_segment', self.is_segment)

        self.patch_length = 32 if "SwinT" in args.backbone else 28
        self.imsize = args.imsize

        self.reg_token = nn.Embedding(1, hidden_dim)
        self.visual_pos = nn.Embedding(self.patch_length ** 2 + 1, hidden_dim)
        self.text_pos = nn.Embedding(args.max_query_len, hidden_dim)
        decoder = dict(
            num_layers=args.vl_enc_layers,
            layer=dict(
                d_model=hidden_dim, nhead=8, dim_feedforward=args.dim_feedforward, dropout=0.1, activation='relu'),
            is_eliminate=args.is_eliminate,
        )
        print("dim_feedforward", args.dim_feedforward)

        from .decoder import TransformerDecoder
        self.decoder = TransformerDecoder(decoder, patch_length=self.patch_length, eliminated_threshold=args.eliminated_threshold)

        if self.is_segment:
            self.mask_head = MLP(hidden_dim, hidden_dim, 256, 2)
            self.mask_cnn = nn.Conv2d(1, 1, 5, padding=2)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        # Vision
        vis_bb = self.visumodel.backbone[0].body
        vis_joiner_pos_emb =  self.visumodel.backbone[1]
        # doing this to follow the orginal implementation in the backbone base forward 
        # The data tensor will get convert back to normal tensor
        if isinstance(img_data, (list, torch.Tensor)):
            nested_img = nested_tensor_from_tensor_list(img_data)
        else:
            nested_img = img_data
        # convert to normal tensor
        img_tensor = nested_img.tensors

        # inside ViTDet
        vis_feats = vis_bb.patch_embed(img_tensor)
        if vis_bb.pos_embed is not None:
            vis_feats = vis_feats + get_abs_pos(
                vis_bb.pos_embed, vis_bb.pretrain_use_cls_token, (vis_feats.shape[1], vis_feats.shape[2])
            )

        # Language
        txt_bb = self.textmodel.bert
        # in BertModel
        txt_tensor  = text_data.tensors
        attention_mask = text_data.mask
        token_type_ids = torch.zeros_like(txt_tensor)
        extended_attention_mask =attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(txt_bb.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        txt_feats = txt_bb.embeddings(txt_tensor, token_type_ids)
     
     
        all_encoder_txt_layers = []
        # visual backbone
        for i in range(self.enc_num):
            

            #Fusion 3 differnt levels cross-output  QKV, O and MLP 
            ####### QKV FUSION
            fusion_vis =  vis_feats.reshape(vis_feats.shape[0], -1, vis_feats.shape[-1]).mean(dim=1) # for text fusion in QKV                
            fusion_txt =  txt_feats.mean(dim=1) # for vision fusion in QKV  

            ## Attention shorcuts
            vis_shortcut = vis_feats.clone()
            txt_shortcut = txt_feats.clone()
            ## Attention
            # Vision
            vis_feats = vis_bb.blocks[i].norm1(vis_feats)
            if vis_bb.blocks[i].window_size > 0:
                vH, vW  = vis_feats.shape[1], vis_feats.shape[2]
                vis_feats, pad_hw = window_partition(vis_feats, vis_bb.blocks[i].window_size)
            vis_feats = qkv_attn_vision(vis_bb.blocks[i].attn, vis_feats, fusion_txt)
            ## Text
            txt_feats = qkv_attn_text(txt_bb.encoder.layer[i].attention.self, txt_feats, fusion_vis, extended_attention_mask)

     
            ###### Out Fusion
            ## Vision
            if vis_bb.blocks[i].window_size > 0:
                fusion_vis2 =  window_unpartition(vis_feats, vis_bb.blocks[i].window_size, pad_hw, (vH, vW))
                fusion_vis2 = fusion_vis2.reshape(fusion_vis2.shape[0], -1, fusion_vis2.shape[-1]).mean(dim=1) 
            else: 
                fusion_vis2 = vis_feats
                fusion_vis2 = fusion_vis2.reshape(fusion_vis2.shape[0], -1, fusion_vis2.shape[-1]).mean(dim=1) 
            fusion_txt2 = txt_feats.mean(dim=1)

            # Out projection
            # vision
            vis_feats = out_attn_vision(vis_bb.blocks[i].attn, vis_feats, fusion_txt2)
            if vis_bb.blocks[i].window_size > 0:
                vis_feats = window_unpartition(vis_feats, vis_bb.blocks[i].window_size, pad_hw, (vH, vW))
            vis_feats = vis_shortcut + vis_bb.blocks[i].drop_path(vis_feats)
            
            ## Text
            txt_feats = out_attn_text(txt_bb.encoder.layer[i].attention.output, txt_feats,fusion_vis2, txt_shortcut)
    

            
            #### MLP Fusion   
            fusion_vis3 =  vis_feats
            fusion_vis3 =  fusion_vis3.reshape(fusion_vis3.shape[0], -1, fusion_vis3.shape[-1]).mean(dim=1)
            fusion_txt3 = txt_feats.mean(dim=1)
            ######  MLP shortcut
            vis_shortcut2 =  vis_feats.clone()
            txt_shortcut2 = txt_feats.clone()
       
            ###  Vision
            vis_feats =  vis_bb.blocks[i].norm2(vis_feats)
            vis_feats = mlp1_vision(vis_bb.blocks[i].mlp, vis_feats, fusion_txt3)
            vis_feats = mlp2_vision(vis_bb.blocks[i].mlp, vis_feats, fusion_txt3)
            vis_feats = vis_shortcut2 + vis_bb.blocks[i].drop_path(vis_feats)
            ######  Text
            txt_feats = mlp1_txt(txt_bb.encoder.layer[i].intermediate, txt_feats, fusion_vis3)
            txt_feats = mlp2_txt(txt_bb.encoder.layer[i].output, txt_feats, fusion_vis3,txt_shortcut2)


            all_encoder_txt_layers.append(txt_feats)
            
            
            

        # Vision
        # inside VitDet
        vis_feats_out = {'0': vis_feats.flatten(1, 2)}
        # convert it vis feature back to Nested Tensor (in Backbone base)
        nested_feats_out: Dict[str, NestedTensor] = {}
        for name, x in vis_feats_out.items():
            m = nested_img.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=self.visumodel.backbone[0].size).to(torch.bool)[0]
            nested_feats_out[name] = NestedTensor(x, mask)
        # Vis (in Joiner)
        joiner_feats: List[NestedTensor] = [] # Feature
        joiner_pos = []
        for name, x in nested_feats_out.items():
            joiner_feats.append(x)
            # position encoding
            joiner_pos.append(vis_joiner_pos_emb(x).to(x.tensors.dtype))
        # in Visual Backbone class forward
        visu_src, visu_mask = joiner_feats[-1].decompose()
        visu_mask = visu_mask.flatten(1)
        visu_src = visu_src.permute(1,0,2)

        
        # Langauge 
        # Bert in modeling
        sequence_output = all_encoder_txt_layers[-1]
        # pooled_out_txt = txt_bb.pooler(sequence_output, vis_feats)
        # Bert in bert.py
        # print(len(all_encoder_txt_layers))
        text_src = all_encoder_txt_layers[self.enc_num-1]
        text_mask = text_data.mask.to(torch.bool)
        text_mask = ~text_mask



        visu_src = self.visu_proj(visu_src)  # (N*B)xC 
        assert text_mask is not None
        text_src = self.text_proj(text_src)
        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        text_pos = self.text_pos.weight.unsqueeze(1).expand(-1, bs, -1)
        visual_pos = self.visual_pos.weight.unsqueeze(1).expand(-1, bs, -1)
        tgt_src = self.reg_token.weight.unsqueeze(1).expand(-1, bs, -1)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
        visu_src = torch.cat([tgt_src, visu_src], dim=0)
        visu_mask = torch.cat([tgt_mask, visu_mask], dim=1)

        if self.is_segment:
            tgt, origin_idx = self.decoder(tgt=visu_src, memory=text_src,
                                           tgt_key_padding_mask=visu_mask,
                                           memory_key_padding_mask=text_mask, pos=text_pos,
                                           query_pos=visual_pos)
            tgt_visu = tgt[1:]

            mask_output = self.mask_head(tgt_visu).permute(1, 0, 2).contiguous()
            origin_idx = origin_idx[:, 1:].unsqueeze(-1)
            origin_idx -= 1
            origin_idx = origin_idx.expand(-1, -1, mask_output.shape[-1]).contiguous()
            tmp_mask_arr = torch.zeros((bs, self.patch_length ** 2, mask_output.shape[-1]),
                                       dtype=mask_output.dtype,
                                       device=mask_output.device)
            mask_output = tmp_mask_arr.scatter(1, origin_idx, mask_output)
            mask_output = mask_output.reshape(bs, self.patch_length, self.patch_length, 16, 16)
            mask_output = mask_output.permute(0, 1, 3, 2, 4).reshape(bs, 1, self.imsize,
                                                                     self.imsize).contiguous()
            mask_output = self.mask_cnn(mask_output).sigmoid()
        else:
            tgt = self.decoder(tgt=visu_src, memory=text_src,
                               tgt_key_padding_mask=visu_mask,
                               memory_key_padding_mask=text_mask, pos=text_pos,
                               query_pos=visual_pos)

        pred_box = self.bbox_embed(tgt[0]).sigmoid()
        if self.is_segment:
            return pred_box, mask_output
        else:
            return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




def qkv_attn_vision(attn_module, vis_feats, txt_feats):
    B, H, W, _ = vis_feats.shape
    qkv = attn_module.qkv(vis_feats, txt_feats).reshape(B, H * W, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
    # qkv = attn_module.qkv(vis_feats).reshape(B, H * W, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * attn_module.num_heads, H * W, -1).unbind(0)
    attn = (q * attn_module.scale) @ k.transpose(-2, -1)
    if attn_module.use_rel_pos:
        attn = add_decomposed_rel_pos(attn, q, attn_module.rel_pos_h, attn_module.rel_pos_w, (H, W), (H, W))
    attn = attn.softmax(dim=-1)
    vis_feats = (attn @ v).view(B, attn_module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

    return  vis_feats

def out_attn_vision(attn_module, vis_feats, txt_feats):
    return attn_module.proj(vis_feats,txt_feats)
    # return attn_module.proj(vis_feats)

def mlp1_vision(mlp_module,vis_feats, txt_feats ):
    vis_feats = mlp_module.fc1(vis_feats,txt_feats)
    # vis_feats = mlp_module.fc1(vis_feats)
    vis_feats = mlp_module.act(vis_feats)
    vis_feats = mlp_module.drop1(vis_feats)

    return vis_feats

def mlp2_vision(mlp_module,vis_feats, txt_feats ):
    vis_feats = mlp_module.norm(vis_feats)

    vis_feats = mlp_module.fc2(vis_feats, txt_feats)
    # vis_feats = mlp_module.fc2(vis_feats)

    vis_feats = mlp_module.drop2(vis_feats)
    return vis_feats


def qkv_attn_text(attn_module, txt_feats, vis_feats, attention_mask):

    mixed_query_layer = attn_module.query(txt_feats, vis_feats)
    mixed_key_layer = attn_module.key(txt_feats,vis_feats)
    mixed_value_layer = attn_module.value(txt_feats,vis_feats) 

    # mixed_query_layer = attn_module.query(txt_feats)
    # mixed_key_layer = attn_module.key(txt_feats)
    # mixed_value_layer = attn_module.value(txt_feats) 


    query_layer = attn_module.transpose_for_scores(mixed_query_layer)
    key_layer = attn_module.transpose_for_scores(mixed_key_layer)
    value_layer = attn_module.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(attn_module.attention_head_size)
    # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = attn_module.dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (attn_module.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return  context_layer

def out_attn_text(attn_module, txt_feats, vis_feats, txt_shortcut):

    hidden_states = attn_module.dense(txt_feats, vis_feats)
    # hidden_states = attn_module.dense(txt_feats)
    hidden_states = attn_module.dropout(hidden_states)
    hidden_states = attn_module.LayerNorm(hidden_states +  txt_shortcut)

    return hidden_states

def mlp1_txt(mlp_module, txt_feats, vis_feats):
    
    hidden_states = mlp_module.dense(txt_feats, vis_feats)
    # hidden_states = mlp_module.dense(txt_feats)
    hidden_states = mlp_module.intermediate_act_fn(hidden_states)

    return hidden_states

def mlp2_txt(mlp_module, txt_feats,vis_feats, txt_shortcut):

    hidden_states = mlp_module.dense(txt_feats, vis_feats)
    # hidden_states = mlp_module.dense(txt_feats)
    hidden_states = mlp_module.dropout(hidden_states)
    hidden_states = mlp_module.LayerNorm(hidden_states +  txt_shortcut)

    return hidden_states

def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)

def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

