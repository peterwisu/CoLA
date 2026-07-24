import argparse
import datetime
import json
import random
import time
import math

import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler

import datasets
import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, validate, clean_cache
import loralib as lora

def param_summary(model):
    """
    For Vision, Text, and Rest groups, print:
    - LoRA param NAMES
    - Other trainable param NAMES
    - Frozen param NAMES
    - 4-line summary (counts & elements)
    Then a top-line summary showing PARAM ELEMENT counts:
    total_trainable, LoRA and Other
    """
    vision = model.module.visumodel.backbone[0].body
    text   = model.module.textmodel.bert

    def _summarize(name, module):
        items = list(module.named_parameters())
        lora = [(n, p) for n, p in items if "lora" in n]
        other = [(n, p) for n, p in items if "lora" not in n and p.requires_grad]
        frozen = [(n, p) for n, p in items if not p.requires_grad]

        # Print names
        print(f"\n=== {name} backbone: LoRA parameters ({len(lora)}) ===")
        for n, _ in lora:     print("   ", n)
        print(f"\n=== {name} backbone: Other trainable ({len(other)}) ===")
        for n, _ in other:    print("   ", n)
        print(f"\n=== {name} backbone: Frozen ({len(frozen)}) ===")
        for n, _ in frozen:   print("   ", n)

        # Element counts
        elems_lora   = sum(p.numel() for _, p in lora)
        elems_other  = sum(p.numel() for _, p in other)
        elems_frozen = sum(p.numel() for _, p in frozen)
        total_train_e = elems_lora + elems_other

        # 4-line summary
        print(f"\n--- {name} backbone SUMMARY ---")
        print(f"LoRA params           : elements={elems_lora:,}")
        print(f"Other trainable params: elements={elems_other:,}")
        print(f"Frozen params         : elements={elems_frozen:,}")
        print(f"TOTAL trainable       : elements={total_train_e:,}\n")

        return total_train_e, elems_lora, elems_other

    def _summarize_rest(name, items):
        lora  = [(n, p) for n, p in items if "lora" in n]
        other = [(n, p) for n, p in items if "lora" not in n and p.requires_grad]
        frozen= [(n, p) for n, p in items if not p.requires_grad]

        print(f"\n=== {name}: LoRA parameters ({len(lora)}) ===")
        for n, _ in lora:     print("   ", n)
        print(f"\n=== {name}: Other trainable ({len(other)}) ===")
        for n, _ in other:    print("   ", n)
        print(f"\n=== {name}: Frozen ({len(frozen)}) ===")
        for n, _ in frozen:   print("   ", n)

        elems_lora   = sum(p.numel() for _, p in lora)
        elems_other  = sum(p.numel() for _, p in other)
        elems_frozen = sum(p.numel() for _, p in frozen)
        total_train_e = elems_lora + elems_other

        print(f"\n--- {name} SUMMARY ---")
        print(f"LoRA params           : elements={elems_lora:,}")
        print(f"Other trainable params: elements={elems_other:,}")
        print(f"Frozen params         : elements={elems_frozen:,}")
        print(f"TOTAL trainable       : elements={total_train_e:,}\n")

        return total_train_e, elems_lora, elems_other

    # Run summaries
    vis_tot_e, vis_lora_e, vis_other_e = _summarize("Vision", vision)
    txt_tot_e, txt_lora_e, txt_other_e = _summarize("Text",   text)

    rest_items = [
        (n, p) for n, p in model.module.named_parameters()
        if "visumodel" not in n and "textmodel" not in n
    ]
    rest_tot_e, rest_lora_e, rest_other_e = _summarize_rest("Rest of model", rest_items)

    # Top-line summary uses ELEMENT counts
    print("--- Top-line trainable PARAM summary (elements) ---")
    print(f"Vision backbone   : total={vis_tot_e:,} (LoRA={vis_lora_e:,}, Other={vis_other_e:,})")
    print(f"Text backbone     : total={txt_tot_e:,} (LoRA={txt_lora_e:,}, Other={txt_other_e:,})")
    print(f"Rest of model     : total={rest_tot_e:,} (LoRA={rest_lora_e:,}, Other={rest_other_e:,})\n")
    print(f"Total trainable parameters: {vis_tot_e + txt_tot_e + rest_tot_e:,}\n")


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visual', default=1e-5, type=float)
    # parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--loss_alpha', default=1.0, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_drop', default=50, type=int)

    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")
    parser.add_argument('--is_eliminate', action='store_true')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='EEVG',
                        help="Name of model to be exploited.")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")

    parser.add_argument('--imsize', default=448, type=int, help='image size')

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)

    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=768, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--eliminated_threshold', default=0.015, type=float)

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='mask_data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='gref_umd', type=str,
                        help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')

    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--is_segment', action='store_true', help='if use segmentation')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--detr_model', default=None, type=str, help='detr model')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrained checkpoint')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    # parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # added
    parser.add_argument('--bb_ckpt_dir', type=str, default='../bb_ckpt/',
                        help='Vision backbone checkpoint path')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--freeze_bb',  default=False, action='store_true', help='Freeze the vision and text backbone')
    parser.add_argument('--use_lora',  default=False, action='store_true', help='Apply LoRA to the model')
    parser.add_argument('--lora_r', default=8, type=int, help='LoRA rank')
    parser.add_argument('--lora_alpha', default=8, type=int, help='LoRA alpha')
    parser.add_argument('--lr_adapter', default=0.1, type=float, help='Adapter learning rate')
    

    return parser


def main(args):
   

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    if args.use_lora:
        print("*"*20)
        print("Using LoRA")
        print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
        print("*"*20)
        if args.lora_r <= 0 or args.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive integers")
    else:
        if args.lora_r > 0 or args.lora_alpha > 0:
            print("*"*20)
            print("LoRA rank and alpha are set to 0, disabling LoRA")
            print("*"*20)
        args.lora_r = 0
        args.lora_alpha = 0

    # build model
    model = build_model(args)
    model.to(device)



    model_without_ddp = model
    if args.distributed:
        if args.use_lora:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    visu_cnn_param = [p for n, p in model_without_ddp.named_parameters() if
                      (("visumodel" in n) and ("backbone" in n) and p.requires_grad)]
    text_tra_param = [p for n, p in model_without_ddp.named_parameters() if (("textmodel" in n) and p.requires_grad)]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if
                  (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]
    

    param_summary(model)
    print("here")
    if args.use_lora:
        # print("using Lora")
        backbone_params = visu_cnn_param + text_tra_param

    if args.use_lora:
        param_list = [
        {"params": rest_param},
        {"params": backbone_params, "lr": args.lr_adapter},
  
        ]
    else:   
        # print("Not using")
        param_list = [
        {"params": rest_param},
        {"params": visu_cnn_param, "lr": args.lr_visual},
        {"params": text_tra_param, "lr": args.lr_bert},
        ]
    # print("here")  
    # print(model)



    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    if args.lr_scheduler == 'poly':
        def lr_func(epoch):
            return (1 - epoch / args.epochs) ** args.lr_power

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'halfdecay':
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 10))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'cosine':
        def lr_func(epoch):
            if epoch < 3:
                return epoch / 10
            else:
                return 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    # build dataset
    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('val', args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.resume:
        if args.use_lora:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_ckpt = checkpoint['model']
            adapter_sd = model_ckpt["lora"]
            decoder_sd = model_ckpt["decoder"]
            # load lora
            model_without_ddp.load_state_dict(adapter_sd, strict=False)
            # load decoder weight
            model_without_ddp.load_state_dict(decoder_sd, strict=False)

            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
        else:

            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrain is not None:

        raise ValueError("Should not be here YEt")
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print('Loading pretrain model from {}'.format(args.pretrain))

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(str(args) + "\n")

    print("Start training")
    scaler = GradScaler()
    start_time = time.time()
    best_accu = 0
    best_mask_iou = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, scaler, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args.loss_alpha,
            epoch == args.start_epoch, accumulation_steps=args.grad_accum_steps
        )
        lr_scheduler.step()
        clean_cache()
        val_stats = validate(args, model, data_loader_val, device)
       
        print(val_stats)
        clean_cache()
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            print("Saving checkpoint to {}".format(checkpoint_paths))
            if val_stats['accu'] > best_accu:
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                best_accu = val_stats['accu']
            if 'mask_miou' in val_stats and val_stats['mask_miou'] > best_mask_iou:
                checkpoint_paths.append(output_dir / 'best_mask_checkpoint.pth')
                best_mask_iou = val_stats['mask_miou']

            for checkpoint_path in checkpoint_paths:
                

                if args.use_lora:
                    # Lora State Dict
                    adapter_sd = lora.lora_state_dict(model_without_ddp, bias='none')
                    rest_params = {
                        n: p
                        for n, p in model_without_ddp.named_parameters()
                        if p.requires_grad
                        and not (n.startswith("visumodel.") or n.startswith("textmodel."))
                    }
                    rest_sd = {name: model_without_ddp.state_dict()[name] for name in rest_params.keys()}
                    
                    print(sum(p.numel() for p in adapter_sd.values()))
                    print(sum(p.numel() for p in rest_sd.values()))                
                    utils.save_on_master({
                        'model' : {"decoder" : rest_sd, "lora" : adapter_sd},
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'val_accu': val_stats['accu']
                    }, checkpoint_path) 

                  
                    
                
                else:

                    utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)
                    
            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EEVG training script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
