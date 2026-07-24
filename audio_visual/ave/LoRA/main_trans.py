

# from __future__ import print_function
import argparse
from base_options import BaseOptions
from gpuinfo import GPUInfo 
import os
from tqdm import tqdm
import loralib as lora
import sys
from argparse import Namespace


fairseq_path = '/path/to/SSLAM/SSLAM_Inference/cloned_fairseq_copy/fairseq/' ## Please update the absolute path to fairseq here

if os.path.exists(fairseq_path):
    sys.path.append(fairseq_path)
    import fairseq
else:
    raise ImportError(f"Fairseq path does not exist: {fairseq_path}. Please update the path to fairseq in the script")

# --- Step 2: Automatically register all custom code in the SSLAM folder ---
user_dir_path = "/path/to/SSLAM"  # Please update the absolute path to the SSLAM folder here
fairseq.utils.import_user_module(Namespace(user_dir=user_dir_path))




args = BaseOptions().parse()

mygpu = GPUInfo.get_info()[0]

gpu_source = {}

if 'N/A' in mygpu.keys():
    for info in mygpu['N/A']:
        if info in gpu_source.keys():
            gpu_source[info] +=1
        else:
            gpu_source[info] =1

for gpu_id in args.gpu:
    gpu_id = str(gpu_id)

    if gpu_id not in gpu_source.keys():
        print('go gpu:', gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        break
    elif gpu_source[gpu_id] < 1:
        print('go gpu:', gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        break

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.cuda.amp import autocast
import random
from dataloader import *
from nets.AVENet import AVENet
from utils.eval_metrics import segment_level, event_level
import pandas as pd
from ipdb import set_trace
import wandb
from PIL import Image
# from criterion import YBLoss,YBLoss2, InfoNCELoss, MaskInfoNCELoss
from einops import rearrange
import certifi
import sys
from torch.cuda import amp

# from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seeds for deterministic results in PyTorch, NumPy, and Python's random module.
    """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_gpu_utilization():
    import subprocess
    try:
        cmd = "nvidia-smi"
        output = subprocess.check_output(cmd, shell=True)
        decoded_output = output.decode('utf-8')
        print(decoded_output, flush=True)
        return decoded_output
    except Exception as e:
        print("Error retrieving nvidia-smi output:", e)
        return None

from typing import Dict, Any
def compare_state_dicts(
    sd1: Dict[str, Any], 
    sd2: Dict[str, Any],
    model_name_1: str = "Original Model",
    model_name_2: str = "Reloaded Model"
) -> bool:
    """
    Compares two state dictionaries and prints a detailed report of differences.

    Returns:
        bool: True if the state dictionaries are identical, False otherwise.
    """
    # Get the set of keys from both dictionaries
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    # Find keys that are not shared between the two dictionaries
    unique_to_1 = keys1 - keys2
    unique_to_2 = keys2 - keys1
    
    # Find keys that are common to both
    common_keys = keys1.intersection(keys2)
    
    mismatched_tensors = []
    # Compare the tensors for each common key
    for key in sorted(list(common_keys)):
        tensor1 = sd1[key]
        tensor2 = sd2[key]
        if not torch.equal(tensor1, tensor2):
            mismatched_tensors.append(key)

    # --- Print the final report ---
    if not unique_to_1 and not unique_to_2 and not mismatched_tensors:
        print("✅ State dictionaries are numerically identical.")
        return True
    else:
        print("❌ State dictionaries are DIFFERENT.")
        if unique_to_1:
            print(f"   - Keys found only in '{model_name_1}': {sorted(list(unique_to_1))}")
        if unique_to_2:
            print(f"   - Keys found only in '{model_name_2}': {sorted(list(unique_to_2))}")
        if mismatched_tensors:
            print(f"   - Tensors with different values: {sorted(mismatched_tensors)}")
        return False

    
def param_summary(model):
    """
    Summarize parameters for an audio-visual model with LoRA applied to backbones.
    Groups: 
      - Visual backbone (“swin”)
      - Audio backbone  (“htsat”)
      - Fusion blocks   (“tpavi”)
      - Rest of model  (everything else)
    For each group, prints:
      - LoRA parameter NAMES
      - Other trainable parameter NAMES
      - Frozen parameter NAMES
      - 4-line summary of element counts
    Finally, prints a top-line summary of trainable parameters (element counts).
    """
    # Collect all named parameters once
    all_items = list(model.named_parameters())

    # Partition into groups by substring in name
    swin_items  = [(n, p) for n, p in all_items if n.startswith("vit_v.")]
    htsat_items = [(n, p) for n, p in all_items if n.startswith("sslam_a.")]
    # tpavi_items = [(n, p) for n, p in all_items if n.startswith("tpavi.")]
    rest_items  = [
        (n, p) for n, p in all_items
        if not n.startswith("vit_v") and not n.startswith("sslam_a.")
    ]

    def _summarize_group(name, items):
        # Separate LoRA vs other trainable vs frozen
        lora   = [(n, p) for n, p in items if "lora" in n.lower()]
        other  = [(n, p) for n, p in items if "lora" not in n.lower() and p.requires_grad]
        frozen = [(n, p) for n, p in items if not p.requires_grad]

        # Print parameter NAMES
        print(f"\n=== {name}: LoRA parameters ({len(lora)}) ===")
        for n, _ in lora:
            print("   ", n)

        print(f"\n=== {name}: Other trainable parameters ({len(other)}) ===")
        for n, _ in other:
            print("   ", n)

        print(f"\n=== {name}: Frozen parameters ({len(frozen)}) ===")
        for n, _ in frozen:
            print("   ", n)

        # Count elements (scalars) in each subset
        elems_lora   = sum(p.numel() for _, p in lora)
        elems_other  = sum(p.numel() for _, p in other)
        elems_frozen = sum(p.numel() for _, p in frozen)
        total_train  = elems_lora + elems_other

        # 4-line summary
        print(f"\n--- {name} SUMMARY ---")
        print(f"LoRA params           : elements={elems_lora:,}")
        print(f"Other trainable params: elements={elems_other:,}")
        print(f"Frozen params         : elements={elems_frozen:,}")
        print(f"TOTAL trainable       : elements={total_train:,}\n")

        return total_train, elems_lora, elems_other

    # Summarize each group
    vis_tot, vis_lora, vis_other   = _summarize_group("Visual backbone (maxvit vision)", swin_items)
    aud_tot, aud_lora, aud_other   = _summarize_group("Audio backbone (maxvit vision)", htsat_items)
    rest_tot, rest_lora, rest_other = _summarize_group("Rest of model", rest_items)

    # Top-line summary
    print("--- Top-line trainable PARAM summary (elements) ---")
    print(f"Visual backbone   : total={vis_tot:,} (LoRA={vis_lora:,}, Other={vis_other:,})")
    print(f"Audio backbone    : total={aud_tot:,} (LoRA={aud_lora:,}, Other={aud_other:,})")
  
    print(f"Rest of model     : total={rest_tot:,} (LoRA={rest_lora:,}, Other={rest_other:,})")
    grand_total = vis_tot + aud_tot  + rest_tot
    print(f"\nTotal trainable parameters: {grand_total:,}\n")

    total_param = sum(i.numel() for i in model.parameters())
    print(f"\nTotal  parameters: {total_param:,}\n")

    print(f"\nUpdate ratio {(grand_total/total_param)*100}")
    exit()


def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    accum_iter = args.accum_itr

    optimizer.zero_grad()
    for batch_idx, sample in enumerate((train_loader)):
        
        audio_spec, gt = sample['audio_spec'].to('cuda'), sample['GT'].to('cuda')
        image = sample['image'].to('cuda')

  
        output = model(audio_spec, image)
        loss = criterion(output.squeeze(1),rearrange(gt, 'b t class -> (b t) class')) 

  
        loss = loss / accum_iter  # normalize the loss to account for gradient accumulation
        loss.backward()
        # weights update
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            # print(batch_idx, flush=True)
            optimizer.step()
            optimizer.zero_grad()

        if batch_idx % 50 == 0:
            # get_gpu_utilization()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_total: {:.6f}'.format(
                epoch, batch_idx * len(audio_spec), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()* accum_iter), flush=True) #  

def eval(model, val_loader, args):

    print("Evaluating model on validation set...")
    model.eval()
    total_acc = 0
    total_num = 0

    for batch_idx, sample in enumerate((val_loader)):
        
        audio_spec, gt= sample['audio_spec'].to('cuda'), sample['GT'].to('cuda')
        image = sample['image'].to('cuda')
        with torch.no_grad():
            output = model(audio_spec, image)
        total_acc += (output.squeeze(1).argmax(dim=-1) == rearrange(gt, 'b t class -> (b t) class').argmax(dim=-1)).sum()
        total_num += output.size(0)

    acc = total_acc/total_num


    # print('val acc: %.2f'%(acc*100))
    print(f'val acc: f{acc}')
    return acc

def main():
    set_seed()
    model = AVENet(args).cuda()
    param_summary(model)

    if args.mode == 'train':

        train_dataset = AVE_dataset(opt = args)
        val_dataset = AVE_dataset(opt = args, mode='test')

        print(" len(train_dataset), len(val_dataset)",len(train_dataset), len(val_dataset))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory = True)


        mlp_params   = []
        lora_params  = []
        mlp_total = 0
        lora_total = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "mlp_class" in name:
                mlp_params.append(param)
                mlp_total += param.numel()
            elif 'lora_' in name:
                # everything else (LoRA, adapters, etc.) goes here
                lora_params.append(param)
                lora_total += param.numel()
            else:
                raise ValueError(f"Unexpected trainable param: {name} - {param.shape}")

        print(f"Total trainable parameters: {mlp_total + lora_total:,} (MLP={mlp_total:,}, LoRA={lora_total:,})")
        # (2) Now build your optimizer with two parameter‐groups:
        optimizer = torch.optim.Adam([
            {"params": mlp_params,  "lr": args.lr_mlp},
            {"params": lora_params, "lr": args.lr},
            ])

        criterion = nn.BCELoss()

        best_F = 0
        count = 0
        for epoch in range(1, args.epochs + 1):


            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            F_event  = eval(model, val_loader, args)
            # F_event2  = eval(model, val_loader, args)
            # print(f"torch equal F1 and F2 : {torch.equal(F_event, F_event2)}")
            count +=1

            # Save as latest checkpoint (always)
            checkpoint_path = f"{args.model_save_dir}{args.checkpoint}_checkpoint.pth"


            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'F_event': F_event,
                'args': args
            }, checkpoint_path)
            print(f"Saved checkpoint to: {checkpoint_path}")

            # If it's the best, also save as best
            if F_event >= best_F:
                count = 0
                best_F = F_event
                print('#################### NEW BEST MODEL #####################')
                print(f"New best F_event: {F_event:.4f}")
                
                best_path = f"{args.model_save_dir}{args.checkpoint}_best_f.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_F': best_F,
                    'args': args
                }, best_path)
                print(f"Saved best model to: {best_path}")
                # print("Verifying saved checkpoint...")
                # # Create a fresh model
                # test_model = AVENet(args).cuda()
                # # Load the checkpoint
                # checkpoint = torch.load(best_path)
                # test_model.eval()
                # # Load state dict with strict=False to detect missing/unexpected keys
                # missing_keys, unexpected_keys = test_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                # print("Missing keys:", missing_keys)
                # print("Unexpected keys:", unexpected_keys)
                # # Create dummy audio tensor: [batch_size, sequence_length, audio_features, time_steps]
                # dummy_audio = torch.randn(2, 10, 1024, 128)
                # # Create dummy visual tensor: [batch_size, sequence_length, channels, height, width]  
                # dummy_visual = torch.randn(2, 10, 3, 224, 224)
                # with torch.no_grad():
                #     dummy_output1 = model(dummy_audio.cuda(), dummy_visual.cuda())
                #     dummy_output2 = test_model(dummy_audio.cuda(), dummy_visual.cuda())
                #     print("Dummy output equal:", torch.equal(dummy_output1, dummy_output2))
                # # Compare parameter values for a few keys
                # print("\nComparing parameter values for first 5 keys:")
                # model_state_dict = model.state_dict()  # original model before saving
                # loaded_state_dict = checkpoint['model_state_dict']
                # for k in model_state_dict.keys():
                #     assert torch.allclose(model_state_dict[k], loaded_state_dict[k])
                # # Evaluate the loaded model
                # F_event_loaded = eval(test_model, val_loader, args)
                # # Compare
                # print(f"Original F_event: {F_event }")
                # print(f"Loaded F_event: {F_event_loaded}")
                # print("outputs equal?",  torch.equal(F_event, F_event_loaded))
            
            
            if count == args.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        print('Training complete best_F: %.4f' % best_F, flush=True)

    elif args.mode == 'eval':
  
        best_path = f"{args.model_save_dir}{args.checkpoint}_best_f.pth"
        best_path = best_path.replace('checkpoints_eval', 'checkpoints')
        print(f"Loading best model from: {best_path}", flush=True)
        test_dataset = AVE_dataset(opt = args, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.eval()
        model.load_state_dict(torch.load(best_path)['model_state_dict'])
        eval(model, test_loader, args)
if __name__ == '__main__':
    main()

