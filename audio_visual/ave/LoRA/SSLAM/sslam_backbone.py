import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf

import torch
import torch.nn.functional as F
import torchaudio
import csv

import os
import sys

fairseq_path = '/path/to/SSLAM/SSLAM_Inference/cloned_fairseq_copy/fairseq/' ## Please update the absolute path to fairseq here

if os.path.exists(fairseq_path):
    sys.path.append(fairseq_path)
    import fairseq
else:
    raise ImportError(f"Fairseq path does not exist: {fairseq_path}. Please update the path to fairseq in the script")



@dataclass
class UserDirModule:
    user_dir: str
    



def get_sslam_backbone(checkpoint_dir=None,   
                        use_lora=True,
                        lora_r=16,
                        lora_alpha=8):
    # hard code for now
    checkpoint_dir = "/path/to/SSLAM_checkpoints/checkpoint_best.pt" ## Please update the absolute path to the SSLAM checkpoint here
    model_dir = "/path/to/SSLAM"  # Please update the absolute path to the SSLAM folder here
    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task_modified([checkpoint_dir], 
                                                                                        use_lora=use_lora,
                                                                                        lora_r=lora_r,
                                                                                        lora_alpha=lora_alpha,
                                                                                        strict=False,
                                                                                        )
    model = model[0]
    # model.eval()
    # model.cuda()
    return model


if __name__ == '__main__':
    BS=3
    device = 'cuda:0'
    x = torch.randn(BS,1,128,128).to(device)
    model = get_sslam_backbone().to(device)
    model.eval()
    y = model(x)
    print(y.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")




