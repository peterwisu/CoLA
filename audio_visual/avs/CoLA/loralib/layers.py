#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from functools import partial
from itertools import repeat
import collections.abc
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result


class CMLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        c_features = None,
        reduction_ratio=0,
        lora_c_scaling = 0,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        if c_features is None:
            raise ValueError("c_feature is None")
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_Am = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_Bm = nn.Parameter(self.weight.new_zeros((out_features, r)))

            self.lora_Ac = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_Bc = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_scaling_m = self.lora_alpha / self.r

            
            # initialize from alpha/r
            self.lora_scaling_c = nn.Parameter(self.weight.new_full((1,), fill_value=lora_c_scaling))
            # print(self.lora_scaling_c)
            # print(reduction_ratio)
    
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.lora_mlp = Mlp(in_features=c_features,
            hidden_features=int(c_features/reduction_ratio),
            out_features=int(self.r *self.r),
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            rank=r)

            

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_Am'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_Am, a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bm)

            nn.init.kaiming_uniform_(self.lora_Ac, a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bc)


    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_Bm @ self.lora_Am) * self.lora_scaling_m
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_Bm @ self.lora_Am) * self.lora_scaling_m
                self.merged = True  
        

    def generate_phi(self, x_c):
        phi_c = self.lora_mlp(x_c)
        return phi_c 

    def forward(self, x_m: torch.Tensor, x_c: torch.Tensor,  partition_feat=False, is_sequence=False):

        if x_c is None:
            raise ValueError("X_c in None in CMLinear")
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            # print("x_m",x_m.shape)
            # print("x_c",x_c.shape) 
            result = F.linear(x_m, T(self.weight), bias=self.bias)   
            # PET (main modality)
            result += (self.lora_dropout(x_m) @ self.lora_Am.transpose(0, 1) @ self.lora_Bm.transpose(0, 1)) * self.lora_scaling_m
            # PEF (cross modality)
            phi_c = self.generate_phi(x_c)
            


            if not is_sequence:
                b, h2, w2, dim= x_m.shape 
                x_m = x_m.reshape(b, -1, dim)

            if partition_feat:
                num_windows_per_batch = x_m.shape[0] // phi_c.shape[0]
                if num_windows_per_batch >1:
                    phi_c = phi_c.repeat_interleave(num_windows_per_batch, dim=0)

            # print("phi_c",phi_c.shape) 
            cross_result = (self.lora_dropout(x_m) @ self.lora_Ac.transpose(0,1) @ phi_c.permute(0,2,1) @ self.lora_Bc.transpose(0,1)) * self.lora_scaling_c
         
            if not is_sequence:
                cross_result= cross_result.reshape(b, h2,w2, -1)

            result += cross_result
            return result
        else:
            # Eval part

            # Normal Lora with Merge weight (Already Merge)
            main  = F.linear(x_m, T(self.weight), bias=self.bias)
            # cross_modal lora (MMLORA)
            # generate phi from cross_modal feature in x_c
            phi_c = self.generate_phi(x_c)


            if not is_sequence:
                b, h2, w2, dim= x_m.shape 
                x_m = x_m.reshape(b, -1, dim)

            if partition_feat:
                num_windows_per_batch = x_m.shape[0] // phi_c.shape[0]
                if num_windows_per_batch >1:
                    phi_c = phi_c.repeat_interleave(num_windows_per_batch, dim=0)
      
            
            # Fusion in MMLora
            cross_result = (x_m @ self.lora_Ac.transpose(0,1) @ phi_c.permute(0,2,1) @ self.lora_Bc.transpose(0,1)) * self.lora_scaling_c
            
            if not is_sequence:
                cross_result= cross_result.reshape(b, h2,w2, -1)
          
            outs = main + cross_result
            return outs


class CMMergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        reduction_ratio=0,
        c_features = None,
        lora_c_scaling = 0,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights,
                           )
        if c_features is None:
            raise ValueError("c_feature is None")
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_Am = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_Bm = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)

            self.lora_Ac = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_Bc = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)

 
            mlp_kwargs = dict(in_features=c_features,
            hidden_features=int(c_features/reduction_ratio),
            out_features=int(self.r*self.r),
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            rank=r
            )
            # 2) ModuleDict approach
            self.lora_mlps = nn.ModuleDict({
                head: Mlp(**mlp_kwargs) for head in ("q", "k", "v")
            })\

   
            self.lora_scaling_m = self.lora_alpha / self.r

            # initilize from alpha / r
            self.lora_scaling_c = nn.Parameter(self.weight.new_full((sum(enable_lora),),fill_value=lora_c_scaling))
            # print(self.lora_scaling_c)
            # print(reduction_ratio)
       
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_Am'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_Bm)
            nn.init.kaiming_uniform_(self.lora_Am, a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bc)
            nn.init.kaiming_uniform_(self.lora_Ac, a=math.sqrt(5))


    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_ABm(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_Am.unsqueeze(0), 
            self.lora_Bm.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))
    
    def get_ABc(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
 
        # B has shape (3*hidden_size, rank) == (2304, 16)
        hidden_size = self.lora_Bc.size(0) // 3  # → 2304//3 == 768
        # Split B into B_q, B_k, B_v each of shape (hidden_size, rank) == (768, 16)
        B_q, B_k, B_v = torch.split(self.lora_Bc, hidden_size, dim=0)
        # (shape (3*rank, in_features)) -> (16,768):
        A_q, A_k, A_v = torch.split(self.lora_Ac,self.r, dim=0)

        ##### !!!!!! Since we not using fan_in_fan_out model T() function is not apply in this case
        return  (B_q, B_k, B_v) ,(A_q, A_k, A_v)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_ABm() * self.lora_scaling_m
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_ABm() * self.lora_scaling_m
                self.merged = True        

    def generate_phi(self, x_c):
        bs = x_c.shape[0]
        phi_c_q = self.lora_mlps['q'](x_c)
        phi_c_k = self.lora_mlps['k'](x_c)
        phi_c_v = self.lora_mlps['v'](x_c)
 
        return (phi_c_q,phi_c_k,phi_c_v) ,bs
    
    def forward(self, x_m: torch.Tensor,  x_c: torch.Tensor, partition_feat=False, is_sequence=False):

        if x_c is None:
            raise ValueError("X_c in None in CMLinear")


        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            main = F.linear(x_m, T(self.weight), bias=self.bias)
            # PET (main modality)
            # PEF (cross modality)
            (phi_c_q,phi_c_k,phi_c_v) ,bs= self.generate_phi(x_c)
            (Bc_q, Bc_k, Bc_v) ,(Ac_q, Ac_k, Ac_v) = self.get_ABc()
            


            if not is_sequence:
                b, h2, w2, dim= x_m.shape 
                x_m = x_m.reshape(b, -1, dim)

            if partition_feat:
                num_windows_per_batch = x_m.shape[0] // phi_c_v.shape[0]
                if num_windows_per_batch >1:
                    phi_c_q = phi_c_q.repeat_interleave(num_windows_per_batch, dim=0)
                    phi_c_k = phi_c_k.repeat_interleave(num_windows_per_batch, dim=0)
                    phi_c_v = phi_c_v.repeat_interleave(num_windows_per_batch, dim=0)

            qc = (self.lora_dropout(x_m) @ Ac_q.transpose(0,1) @ phi_c_q.permute(0,2,1) @ Bc_q.transpose(0,1)) * self.lora_scaling_c[0]
            kc = (self.lora_dropout(x_m) @ Ac_k.transpose(0,1) @ phi_c_k.permute(0,2,1) @ Bc_k.transpose(0,1)) * self.lora_scaling_c[1]
            vc = (self.lora_dropout(x_m) @ Ac_v.transpose(0,1) @ phi_c_v.permute(0,2,1) @ Bc_v.transpose(0,1)) * self.lora_scaling_c[2]

            cross_result = torch.cat([qc,kc,vc],dim=-1)
            
            if not is_sequence:
                cross_result= cross_result.reshape(b, h2,w2, -1)

            outputs = main + cross_result

            return outputs
        else:
            result = F.linear(x_m, T(self.weight), bias=self.bias)
            if self.r > 0:

                # PET (main modality)
                result += self.lora_dropout(x_m) @ T(self.merge_ABm().T) * self.lora_scaling_m
                # PEF (cross modality)
                (phi_c_q,phi_c_k,phi_c_v) ,bs= self.generate_phi(x_c)
                (Bc_q, Bc_k, Bc_v) ,(Ac_q, Ac_k, Ac_v) = self.get_ABc()
                

                if not is_sequence:
                
                    b, h2, w2, dim= x_m.shape 
                    x_m = x_m.reshape(b, -1, dim)

                
                if partition_feat:
                    num_windows_per_batch = x_m.shape[0] // phi_c_v.shape[0]
                    if num_windows_per_batch >1:
                        phi_c_q = phi_c_q.repeat_interleave(num_windows_per_batch, dim=0)
                        phi_c_k = phi_c_k.repeat_interleave(num_windows_per_batch, dim=0)
                        phi_c_v = phi_c_v.repeat_interleave(num_windows_per_batch, dim=0)

                # print(x_m.shape)
                # print(phi_c_q.shape)
                qc = (self.lora_dropout(x_m) @ Ac_q.transpose(0,1) @ phi_c_q.permute(0,2,1) @ Bc_q.transpose(0,1)) *  self.lora_scaling_c[0]
                kc = (self.lora_dropout(x_m) @ Ac_k.transpose(0,1) @ phi_c_k.permute(0,2,1) @ Bc_k.transpose(0,1)) *  self.lora_scaling_c[1]
                vc = (self.lora_dropout(x_m) @ Ac_v.transpose(0,1) @ phi_c_v.permute(0,2,1) @ Bc_v.transpose(0,1)) *  self.lora_scaling_c[2]

                cross_result = torch.cat([qc,kc,vc],dim=-1)

                if not is_sequence:
                    cross_result= cross_result.reshape(b, h2,w2, -1)


                result += cross_result

            return result




class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            rank=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        linear_layer =  nn.Linear #nn.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) #if norm_layer is not None else nn.Identity()
        self.drop2 = nn.Dropout(drop_probs[1])
        self.norm2 = norm_layer([rank,rank])
        self.rank = rank

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)

        x = x.reshape(x.shape[0],self.rank,self.rank)
        x = self.norm2(x) # add this LN in this experiment

        return x
    
