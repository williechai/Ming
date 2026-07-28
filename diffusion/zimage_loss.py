from accelerate import scheduler
from transformers import CLIPTokenizer, CLIPTextModel, SiglipModel
import torch
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from tqdm import tqdm
from typing import Any, Mapping

import math
import copy
# import atorch

import torchvision
from diffusers import AutoencoderKL

from IPython import embed
import argparse
import gc
import json
import os
import random
import threading

from collections import OrderedDict

import diffusers
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
)
#from .sd3_transformer import SD3Transformer2DModel
from .transformer_z_image import ZImageTransformer2DModel
from .pipeline_z_image import ZImagePipeline
import torch.nn.functional as F
import torch.nn as nn
from .autoencoder_kl_qwenimage import AutoencoderKLQwenImage

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToClipMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(input_dim, 2048)
        self.layer_norm1 = nn.LayerNorm(2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states

class ZImageModel_withMLP(nn.Module):
    def __init__(self, transformer, vision_dim=1152, use_identity_mlp=False, text_encoder_norm=False):
        super().__init__()
        self.transformer = transformer
        self.dtype = torch.bfloat16
        self.mlp = ToClipMLP(vision_dim, 2560) if not use_identity_mlp else nn.Identity()
        # self.mlp_pool = ToClipMLP(vision_dim, 768)
        self.config = self.transformer.config
        self.in_channels = self.transformer.in_channels
        self.text_encoder_norm = text_encoder_norm

        # 需要搭配使用
        #if text_encoder_norm or use_identity_mlp:
        #    assert use_identity_mlp and text_encoder_norm

    
    def forward(self, hidden_states,
                    timestep,
                    encoder_hidden_states,
                    return_dict,
                    encoder_attention_mask=None,
                    extra_vit_input=None,
                    ref_hidden_states=None,
                    encoder_hidden_states_2=None,
                     **kargs):

        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)

            if self.text_encoder_norm:
                encoder_hidden_states = F.normalize(encoder_hidden_states, dim=-1) * 1000.0 # 1000是原始text_encoder的norm
            
            encoder_hidden_states = self.mlp(encoder_hidden_states)
            
            if extra_vit_input is not None:
                encoder_hidden_states = torch.cat((encoder_hidden_states, extra_vit_input), dim=1)

            encoder_hidden_states = list(encoder_hidden_states.unbind(dim=0))
            
        hidden_states = self.transformer(
                    x=hidden_states,
                    cap_feats=encoder_hidden_states,
                    t=timestep,
                    return_dict=False,
                    ref_x=ref_hidden_states,
                    cap_feats_2=encoder_hidden_states_2,
                     **kargs
                )
        return hidden_states

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

def latent_mean_variance(latents: torch.Tensor, per_channel: bool = True, unbiased_var: bool = False):
    """
    统计 latent 的均值与方差（与扩散训练使用的 ``latents`` 形状一致，一般为 ``[B, C, H, W]``）。
    Args:
        latents: ``[B, C, H, W]``；若为 ``[B, C, T, H, W]`` 且 ``T==1`` 会先 ``squeeze(2)``。
        per_channel: True 时在 ``(B, H, W)`` 上聚合，得到每个通道一个标量，shape ``[C]``；
            False 时对整个 tensor 一个全局均值、方差。
        unbiased_var: 是否使用无偏方差（Bessel）。
    Returns:
        ``(mean, variance)``；标准差为 ``variance.sqrt()``（注意数值稳定性可用 ``torch.sqrt(variance.clamp_min(0))``）。
    """
    z = latents
    if z.dim() == 5 and z.shape[2] == 1:
        z = z.squeeze(2)
    if z.dim() != 4:
        raise ValueError(f"Expected 4D latents [B,C,H,W] (or 5D with T=1), got {tuple(z.shape)}")
    if per_channel:
        dims = (0, 2, 3)
        mean_v = z.mean(dim=dims)
        var_v = z.var(dim=dims, unbiased=unbiased_var)
    else:
        mean_v = z.mean()
        var_v = z.var(unbiased=unbiased_var)
    return mean_v, var_v

class ZImageLoss(torch.nn.Module):
    def __init__(self, 
            model_path, 
            vision_dim=2560, 
            scheduler_path=None,
            mlp_state_dict=None,
            torch_dtype=torch.float32,
            device='cpu',
            use_identity_mlp=False,
            text_encoder_norm=False,
        ):
        super(ZImageLoss, self).__init__()

        if device is not None:
            self.device = torch.device(device)   
        else:
            self.device = torch.device(torch.cuda.current_device())    

        self.scheduler_path = scheduler_path

        vae_config_path = os.path.join(model_path, "vae", "config.json")
        assert os.path.exists(vae_config_path)

        with open(vae_config_path, "r") as f:
            vae_config =  json.load(f)
            if "_class_name" in vae_config and vae_config["_class_name"] == "AutoencoderKLQwenImage":
                self.vae = AutoencoderKLQwenImage.from_pretrained(
                    model_path,
                    subfolder="vae",
                    torch_dtype=torch_dtype,
                )
                self.vae_sample_mode = 'argmax'
            else:
                self.vae = AutoencoderKL.from_pretrained(
                    model_path,
                    subfolder="vae",
                    torch_dtype=torch_dtype,
                )
                self.vae_sample_mode = 'sample'
        
        self.vae.input_channels = 4 if ('input_channels' in self.vae.config and self.vae.config.input_channels == 4) or ('in_channels' in self.vae.config and self.vae.config.in_channels == 4) else 3
        
        # self.vae.to(self.torch_type).to(self.device)
        self.vae.requires_grad_(False)

        self.train_model = ZImageTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer",
            torch_dtype=torch_dtype,
        )

        self.train_model = ZImageModel_withMLP(self.train_model, vision_dim=vision_dim, use_identity_mlp=use_identity_mlp, text_encoder_norm=text_encoder_norm)

        assert mlp_state_dict is not None
        self.train_model.mlp.load_state_dict(mlp_state_dict, strict=True)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.scheduler_path, subfolder="scheduler")
        self.noise_scheduler.config['use_dynamic_shifting'] = True

        self.pipelines = ZImagePipeline(
            vae=self.vae,
            transformer=self.train_model,  
            text_encoder=None, 
            tokenizer=None,                          
            scheduler=self.noise_scheduler,
        ).to(self.device)

    def set_trainable_params(self, trainable_params):
        
        self.vae.requires_grad_(False)

        if trainable_params == 'all':
            self.train_model.requires_grad_(True)
        else:
            self.train_model.requires_grad_(False)
            for name, module in self.train_model.named_modules():
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        for params in module.parameters():
                            params.requires_grad = True

        num_parameters_trainable = 0
        num_parameters = 0
        name_parameters_trainable = []
        for n, p in self.train_model.named_parameters():
            num_parameters += p.data.nelement()
            if not p.requires_grad:
                continue  # frozen weights
            name_parameters_trainable.append(n)
            num_parameters_trainable += p.data.nelement()
        logger.info(f"number of all Diffusion parameters: {num_parameters}, trainable: {num_parameters_trainable}")
    

    def sample(self, encoder_hidden_states, steps=20, cfg=7.0, image_cfg=1.0, cfg_mode=1, seed=42, height=512, width=512, use_dynamic_shifting=False, extra_vit_input=None, ref_x=None, directvlm_hidden_states=None):
        negative_prompt_embeds = None
        if encoder_hidden_states is not None:
            encoder_hidden_states = list(encoder_hidden_states.unbind(dim=0))
            negative_prompt_embeds= [en * 0 for en in encoder_hidden_states]
        
        encoder_hidden_states_2 = directvlm_hidden_states
        negative_prompt_embeds_2 = None
        if encoder_hidden_states_2 is not None:
            encoder_hidden_states_2 = list(encoder_hidden_states_2.unbind(dim=0))
            negative_prompt_embeds_2= [en * 0 for en in encoder_hidden_states_2]

        image = self.pipelines(
            prompt_embeds=encoder_hidden_states,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_2=encoder_hidden_states_2,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            guidance_scale=cfg,
            #image_guidance_scale=image_cfg,
            #guidance_scale_mode=cfg_mode,
            generator=torch.manual_seed(seed),
            num_inference_steps=steps,
            height=height,
            width=width,
            max_sequence_length=512,
            device=self.device,
            #extra_vit_input=extra_vit_input,
            ref_hidden_states=ref_x,
            #use_dynamic_shifting=use_dynamic_shifting,
            sample_mode='argmax',
        ).images

        return image  
