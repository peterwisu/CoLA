import os
import time
import random
import shutil
import torch
import numpy as np
import argparse


import sys
from argparse import Namespace
from gpuinfo import GPUInfo 
from base_options import BaseOptions


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

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seeds for deterministic results in PyTorch, NumPy, and Python's random module.
    """
    # 1. Set seed for Python's built-in random module
    random.seed(seed)
    
    # 2. Set seed for NumPy
    np.random.seed(seed)  # <-- This line handles NumPy

    # 3. Set seed for PyTorch on both CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4. Configure PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

import logging

from config import cfg
from dataloader import S4Dataset

from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
import certifi
import sys

from tqdm import tqdm
from model import PVT_AVSModel as AVSModel
os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())


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
    swin_items  = [(n, p) for n, p in all_items if n.startswith("swin_v.")]
    htsat_items = [(n, p) for n, p in all_items if n.startswith("audio_a.")]
    # tpavi_items = [(n, p) for n, p in all_items if n.startswith("tpavi.")]
    rest_items  = [
        (n, p) for n, p in all_items
        if not n.startswith("swin_v") and not n.startswith("audio_a.")
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






if __name__ == "__main__":
	# parser = argparse.ArgumentParser()    
	set_seed()
	# args = parser.parse_args()

	# Fix seed
	FixSeed = 123
	random.seed(FixSeed)
	np.random.seed(FixSeed)
	torch.manual_seed(FixSeed)
	torch.cuda.manual_seed(FixSeed)

	# Log directory
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir, exist_ok=True)
	# Logs
	prefix = args.session_name
	# log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
	# args.log_dir = log_dir
	log_dir = args.output_dir +'train_logs'
	print('==> Log directory: {}'.format(log_dir))

	# Save scripts
	script_path = os.path.join(log_dir, 'scripts')
	if not os.path.exists(script_path):
		os.makedirs(script_path, exist_ok=True)

	scripts_to_save = ['train.sh', 'train.py', 'test.sh', 'test.py', 'config.py', 'dataloader.py', './model/PVT_AVSModel.py', 'loss.py']
	for script in scripts_to_save:
		dst_path = os.path.join(script_path, script)
		try:
			shutil.copy(script, dst_path)
		except IOError:
			os.makedirs(os.path.dirname(dst_path), exist_ok=True)
			shutil.copy(script, dst_path)

	# Checkpoints directory
	checkpoint_dir = os.path.join(log_dir, 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir, exist_ok=True)
	args.checkpoint_dir = checkpoint_dir

	# Set logger
	log_path = os.path.join(log_dir, 'log')
	if not os.path.exists(log_path):
		os.makedirs(log_path, exist_ok=True)

	setup_logging(filename=os.path.join(log_path, 'log.txt'))
	logger = logging.getLogger(__name__)
	logger.info('==> Config: {}'.format(cfg))
	logger.info('==> Arguments: {}'.format(args))
	logger.info('==> Experiment: {}'.format(args.session_name))

	# Model
	model = AVSModel.Pred_endecoder(channel=256, \
										opt=args, \
										config=cfg, \
										tpavi_stages=args.tpavi_stages, \
										tpavi_vv_flag=args.tpavi_vv_flag, \
										tpavi_va_flag=args.tpavi_va_flag)

	model = torch.nn.DataParallel(model).cuda()
	model.train()

	param_summary(model.module)

	# video backbone
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# yb:add
	# audio_backbone = torch.nn.DataParallel(audio_backbone).cuda() 

	# Data
	train_dataset = S4Dataset('train', args)
	train_dataloader = torch.utils.data.DataLoader(train_dataset,
														batch_size=args.train_batch_size,
														shuffle=True,
														num_workers=args.num_workers,
														pin_memory=True)
	max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

	val_dataset = S4Dataset('test',args)
	val_dataloader = torch.utils.data.DataLoader(val_dataset,
														batch_size=args.val_batch_size,
														shuffle=False,
														num_workers=args.num_workers,
														pin_memory=True)

	# Optimizer
	model_params = model.parameters()
	optimizer = torch.optim.Adam(model_params, args.lr)


	avg_meter_total_loss = pyutils.AverageMeter('total_loss')
	avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
	avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
	avg_meter_miou = pyutils.AverageMeter('miou')
	# avg_meter_miou_loaded = pyutils.AverageMeter('miou')

	# Train
	best_epoch = 0
	global_step = 0
	miou_list = []
	max_miou = 0
	accum_iter = args.grad_accum

	for epoch in range(args.max_epoches):
		optimizer.zero_grad() 
		for n_iter, batch_data in enumerate((train_dataloader)):
			# imgs, audio, mask = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
			imgs, audio_spec, audio, mask = batch_data

			imgs = imgs.cuda()
			audio = audio.cuda()
			mask = mask.cuda()
			B, frame, C, H, W = imgs.shape
			# imgs = imgs.view(B*frame, C, H, W)
			mask = mask.view(B, H, W)

			audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4]) # [B*T, 1, 96, 64]

			output, visual_map_list, a_fea_list = model(imgs, audio_spec) # [bs*5, 1, 224, 224]
			loss, loss_dict = IouSemanticAwareLoss(output, mask.unsqueeze(1).unsqueeze(1), \
												a_fea_list, visual_map_list, \
												lambda_1=args.lambda_1, \
												count_stages=args.sa_loss_stages, \
												sa_loss_flag=args.sa_loss_flag, \
												mask_pooling_type=args.mask_pooling_type)


			avg_meter_total_loss.add({'total_loss': loss.item()})
			avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
			avg_meter_sa_loss.add({'sa_loss': loss_dict['sa_loss']})

			# original
			# optimizer.zero_grad()
			# loss.backward()
			# optimizer.step()

			loss = loss / accum_iter  # Normalize the loss by the accumulation steps
			loss.backward()

			# weights update
			# if ((n_iter + 1) % accum_iter == 0) or (n_iter + 1 == len(train_dataloader)):
			if (global_step +1 ) % accum_iter ==0:
				# print(batch_idx, flush=True)
				# print(f"Updating weights at step {(global_step + 1)} and iter {n_iter+1}" , flush=True)
				optimizer.step()
				optimizer.zero_grad()  # Reset gradients after updating weights


			if (global_step+1) %accum_iter == 0:
				if (global_step+1) % 100 == 0:
					get_gpu_utilization()
				train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lambda_1:%.4f, lr: %.6f'%(
							global_step+1, 
							max_step, 
							avg_meter_total_loss.pop('total_loss'), 
							avg_meter_iou_loss.pop('iou_loss'), 
							avg_meter_sa_loss.pop('sa_loss'), 
							args.lambda_1, 
							optimizer.param_groups[0]['lr'])
				logger.info(train_log)
			
			global_step += 1
		if global_step % accum_iter != 0:
			# print(f"Finalizing weights update at step {global_step} (not a multiple of {accum_iter})", flush=True)
			optimizer.step()
			optimizer.zero_grad()


		# Validation:
		count = 0 
		model.eval()
		print('==> Start validation at epoch %d'%epoch)
		with torch.no_grad():
			for n_iter, batch_data in enumerate(val_dataloader):
				# imgs, audio, mask, _, _ = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
				imgs, audio_spec, audio, mask,_,_ = batch_data
  

				imgs = imgs.cuda()
				audio = audio.cuda()
				mask = mask.cuda()
				B, frame, C, H, W = imgs.shape
				# imgs = imgs.view(B*frame, C, H, W)
				mask = mask.view(B*frame, H, W)
				# audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
				# with torch.no_grad():
				#     audio_feature = audio_backbone(audio)

				# # ------> yb: add
				# audio_feature = audio_feature.view(B,5, audio_feature.size(-1))
				# # <------

				output, _, _ = model(imgs, audio_spec) # [bs*5, 1, 224, 224]


				miou = mask_iou(output.squeeze(1), mask)
				avg_meter_miou.add({'miou': miou})

			miou = (avg_meter_miou.pop('miou'))

			count = count +1
			if miou > max_miou:
				model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
				torch.save(model.module.state_dict(), model_save_path)
				torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': miou,
                    'args': args
                }, model_save_path)
				best_epoch = epoch
				logger.info('save best model to %s'%model_save_path)
				count = 0

				# print("Verifying saved checkpoint...")
                # # Create a fresh model
				# test_model = AVSModel.Pred_endecoder(channel=256, \
				# 									opt=args, \
				# 									config=cfg, \
				# 									tpavi_stages=args.tpavi_stages, \
				# 									tpavi_vv_flag=args.tpavi_vv_flag, \
				# 									tpavi_va_flag=args.tpavi_va_flag)
				# print(model_save_path, flush=True)	
				# checkpoint = torch.load(model_save_path, map_location='cpu')
				# # test_model.load_state_dict(checkpoint, strict=True)
				# # test_model = torch.nn.DataParallel(test_model).cuda()
				# test_model.eval()
				# # Load state dict with strict=False to detect missing/unexpected keys
				# missing_keys, unexpected_keys = test_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
				# print("Missing keys:", missing_keys)
				# print("Unexpected keys:", unexpected_keys)
				# test_model = torch.nn.DataParallel(test_model).cuda()
                # # # Compare parameter values for a few keys
                # # print("\nComparing parameter values for first 5 keys:")
                # # model_state_dict = model.state_dict()  # original model before saving
                # # loaded_state_dict = checkpoint['model_state_dict']
                # # for k in model_state_dict.keys():
                # #     assert torch.allclose(model_state_dict[k], loaded_state_dict[k])
                # # Evaluate the loaded model
				# test_model.eval()
				# with torch.no_grad():
				# 	for n_iter, batch_data in enumerate(val_dataloader):
				# 		imgs, audio_spec, audio, mask,_,_ = batch_data
				# 		imgs = imgs.cuda()
				# 		audio = audio.cuda()
				# 		mask = mask.cuda()
				# 		B, frame, C, H, W = imgs.shape
				# 		mask = mask.view(B*frame, H, W)
				# 		output, _, _ =test_model(imgs, audio_spec) # [bs*5, 1, 224, 224]
				# 		loaded_miou = mask_iou(output.squeeze(1), mask)
				# 		avg_meter_miou_loaded.add({'miou': loaded_miou})
				# 	loaded_miou = (avg_meter_miou_loaded.pop('miou'))
                # # Compare
				# print(f"Original F_event: {miou}",flush=True)
				# print(f"Loaded F_event: {loaded_miou}", flush=True)
				# print("outputs equal?",  torch.equal(miou, loaded_miou), flush=True)
				# exit()


			if count == args.early_stop:
				logger.info('Early stop at epoch %d'%epoch)
				break

			miou_list.append(miou)
			max_miou = max(miou_list)
			val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
			logger.info(val_log)

		model.train()
	logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))











