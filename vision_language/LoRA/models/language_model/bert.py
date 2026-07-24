# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
from torch import nn
from utils.misc import NestedTensor
# from pytorch_pretrained_bert.modeling import BertModel

from .modeling import BertModel
import os

class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num, bb_ckpt_dir: str, use_lora: bool, lora_r: int, lora_alpha: int):
        super().__init__()
        
        model_ckpt_path = f"{bb_ckpt_dir}/{name}/"
        
        os.makedirs(model_ckpt_path, exist_ok=True)

        print(f"Loading BERT model from : {model_ckpt_path}")
        if name == 'bert-base-uncased':
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num

        self.bert = BertModel.from_pretrained(
                                            name,
                                            cache_dir=model_ckpt_path,
                                            use_lora=use_lora,
                                            lora_r=lora_r,
                                            lora_alpha=lora_alpha,
                                            )

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

        cur_bert_layer_num = len(self.bert.encoder.layer)
        for ind in range(cur_bert_layer_num, 0, -1):
            if ind > self.enc_num:
                del self.bert.encoder.layer[ind - 1]
            else:
                break


    def forward(self, tensor_list: NestedTensor):
        if self.enc_num > 0:
            all_encoder_layers, _ = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
            # use the output of the X-th transformer encoder layers
            xs = all_encoder_layers[self.enc_num - 1]
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out


def build_bert(args):
    train_bert = args.lr_bert > 0
    print(f"Bert model: {args.bert_model}, train_bert: {train_bert}")
    bert = BERT(args.bert_model, 
                train_bert, 
                args.hidden_dim, 
                args.max_query_len, 
                args.bert_enc_num, 
                args.bb_ckpt_dir, 
                args.use_lora, 
                args.lora_r, 
                args.lora_alpha)
    # print(bert)
    # # print(bert.config)
    # exit()
    return bert
