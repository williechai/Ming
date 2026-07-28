import torch
from bisect import bisect_left
import os

from transformers import (
    AutoProcessor,
)

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

import warnings

warnings.filterwarnings("ignore")
from PIL import Image


def split_model():
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = 32
    layer_per_gpu = num_layers // world_size
    layer_per_gpu = [i * layer_per_gpu for i in range(1, world_size + 1)]
    for i in range(num_layers):
        device_map[f'model.model.layers.{i}'] = bisect_left(layer_per_gpu, i)

    device_map['vision'] = 0
    device_map['audio'] = 0
    device_map['linear_proj'] = 0
    device_map['linear_proj_audio'] = 0
    device_map['model.model.word_embeddings.weight'] = 0
    device_map['model.model.norm.weight'] = 0
    device_map['model.lm_head.weight'] = 0
    device_map['model.model.norm'] = 0
    device_map[f'model.model.layers.{num_layers - 1}'] = 0
    return device_map


if __name__ == '__main__':
    model_name_or_path = model_path = os.getenv("MODEL_PATH", './')
    code_path = "."

    processor = AutoProcessor.from_pretrained(code_path, trust_remote_code=True)
    
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=split_model(),
        load_image_gen=True,
    ).to(dtype=torch.bfloat16)

    
    image_save_path_t2i = "./miniv2_bookstore_t2i.jpg"
    
    instruction = "A whimsical comic-style illustration of a cozy bookstore entrance on a sunny afternoon. The storefront features warm brick walls and large glass windows filled with stacked books and potted ferns. Above the wooden door hangs a hand-painted signboard with bold, stylized Chinese characters reading “理解与生成统一” accented with curling vines and tiny stars. Sunlight casts playful shadows on the cobblestone path leading to the door, where a vintage lantern in a sunbeam add charm. The linework is clean, colors vibrant yet soft, evoking a friendly, storybook atmosphere. No people or vehicles are present, emphasizing quiet serenity."
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": instruction},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        image_gen_highres=2048,  
        image_gen_aspect_ratio=1.0,
    ).to(model.device)

    for k in inputs.keys():
        if k in ["pixel_values", "pixel_values_videos", "audio_feats", "image_gen_pixel_values_reference"]:
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    # set `image_gen=True` to enable image generation
    image = model.generate(
        **inputs,
        image_gen=True,
        image_gen_seed=0,
        image_gen_cfg=2,
        image_gen_image_cfg=1.0,
    )

    image.save(image_save_path_t2i)

    image_save_path_i2i_0 = "./miniv2_bookstore_t2i_then_edit.jpg"
    instruction = "add a cat on the ground"

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": image_save_path_t2i},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        image_gen_highres=2048,  
        image_gen_ref_images=Image.open(image_save_path_t2i).convert("RGB"),
    ).to(model.device)


    for k in inputs.keys():
        if k in ["pixel_values", "pixel_values_videos", "audio_feats", "image_gen_pixel_values_reference"]:
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    # set `image_gen=True` to enable image generation
    image = model.generate(
        **inputs,
        image_gen=True,
        image_gen_seed=42,
        image_gen_cfg=2.0,
        image_gen_image_cfg=1.0,
    )

    image.save(image_save_path_i2i_0)
