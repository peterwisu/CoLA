import custom_timm
import torch


def get_timm_EVA_pretrained_model(n_classes=527, imgnet=True, use_lora=False, lora_r=0, lora_alpha=0, lora_c_dim=None, lora_fusion_layers=None, lora_reduction=0, lora_c_scaling=0):
    print(f"\n loading EVA-Lwith imgnet pretrained {imgnet}\n")
    kwargs = {"use_lora": use_lora ,"lora_r": lora_r, "lora_alpha": lora_alpha, "lora_c_dim": lora_c_dim, "lora_fusion_layers" : lora_fusion_layers, "lora_reduction" : lora_reduction, "lora_c_scaling" : lora_c_scaling} 
    m = custom_timm.create_model(model_name='eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=imgnet, pretrained_strict=False ,global_pool='', class_token=True,dynamic_img_size=True,**kwargs)
    # m = custom_timm.create_model(model_name='vit_large_patch14_dinov2.lvd142m', pretrained=imgnet, pretrained_strict=False ,global_pool='', class_token=True,dynamic_img_size=True,**kwargs)
    return m

def get_timm_dino_l_pretrained_model(n_classes=527, imgnet=True, use_lora=False, lora_r=0, lora_alpha=0, lora_c_dim=None, lora_fusion_layers=None, lora_reduction=0, lora_c_scaling=0):
    print(f"\n loading DINOv2-L with imgnet pretrained {imgnet}\n")
    kwargs = {"use_lora": use_lora ,"lora_r": lora_r, "lora_alpha": lora_alpha, "lora_c_dim": lora_c_dim, "lora_fusion_layers" : lora_fusion_layers, "lora_reduction" : lora_reduction, "lora_c_scaling" : lora_c_scaling} 
    m = custom_timm.create_model(model_name='vit_large_patch14_dinov2.lvd142m', pretrained=imgnet, pretrained_strict=False ,global_pool='', class_token=True,dynamic_img_size=True,**kwargs)
    return m

def get_timm_dino_b_pretrained_model(n_classes=527, imgnet=True, use_lora=False, lora_r=0, lora_alpha=0, lora_c_dim=None, lora_fusion_layers=None, lora_reduction=0, lora_c_scaling=0):
    print(f"\n loading DINO-B with imgnet pretrained {imgnet}\n")
    kwargs = {"use_lora": use_lora ,"lora_r": lora_r, "lora_alpha": lora_alpha, "lora_c_dim": lora_c_dim, "lora_fusion_layers" : lora_fusion_layers, "lora_reduction" : lora_reduction, "lora_c_scaling" : lora_c_scaling} 
    m = custom_timm.create_model(model_name='vit_base_patch14_dinov2.lvd142m', pretrained=imgnet, pretrained_strict=False ,global_pool='', class_token=True,dynamic_img_size=True,**kwargs)
    return m

def get_timm_vit_b_pretrained_model(n_classes=527, imgnet=True, use_lora=False, lora_r=0, lora_alpha=0, lora_c_dim=None, lora_fusion_layers=None, lora_reduction=0, lora_c_scaling=0):
    print(f"\n loading ViT-B with imgnet pretrained {imgnet}\n")
    kwargs = {"use_lora": use_lora ,"lora_r": lora_r, "lora_alpha": lora_alpha, "lora_c_dim": lora_c_dim, "lora_fusion_layers" : lora_fusion_layers, "lora_reduction" : lora_reduction, "lora_c_scaling" : lora_c_scaling} 
    m = custom_timm.create_model(model_name='vit_base_patch16_224.augreg_in21k', pretrained=imgnet, pretrained_strict=False ,global_pool='', class_token=True,dynamic_img_size=True,**kwargs)
    return m


