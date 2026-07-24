import custom_timm

def get_timm_swinv2_pretrained_model(n_classes=527, imgnet=True, use_lora=False, lora_r=0, lora_alpha=0, lora_c_dim=None, lora_fusion_layers=None, lora_reduction=0, lora_c_scaling=0):
	print(f"\n loading with imgnet pretrained {imgnet}\n")
	kwargs = {"use_lora": use_lora ,"lora_r": lora_r, "lora_alpha": lora_alpha, "lora_c_dim": lora_c_dim, "lora_fusion_layers" : lora_fusion_layers, "lora_reduction" : lora_reduction, 'lora_c_scaling' : lora_c_scaling} 
	m = custom_timm.create_model('swinv2_large_window12_192_22k', pretrained=imgnet, **kwargs) 
	return m

