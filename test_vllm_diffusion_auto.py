import os
import torch

from transformers import AutoProcessor, GenerationConfig
#from modeling_bailingmm import BailingMMNativeForConditionalGeneration

from IPython import embed
from PIL import Image

import os
os.environ["VLLM_USE_V1"] = "0"

import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt as LLMInputs
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
import sys
import os
import torch
from IPython import embed

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from modeling_bailingmm import BailingMMNativeForConditionalGeneration


class MingOmniImageGen(object):
    def __init__(
        self,
        bl_mm,
    ):
        os.environ["IMAGE_GEN_MODE"] = "None"
        self.processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
        self.model_vllm = LLM(model=bl_mm, trust_remote_code=True, enforce_eager=True, disable_custom_all_reduce=True, tensor_parallel_size=1, limit_mm_per_prompt={"image": 10})

        # Load pre-trained model with optimized settings, this will take ~10 minutes
        self.model_diffusion = BailingMMNativeForConditionalGeneration.from_pretrained(
            "inclusionAI/Ming-Lite-Omni",
            torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
            attn_implementation="flash_attention_2",
            load_image_gen=True,
            low_cpu_mem_usage=True,       # Minimize CPU memory during loading
            load_vlm=False,               # No vlm, only diffusion
        ).to("cuda").to(torch.bfloat16)                     # Run on GPU

    
    def image_gen(self, prompt, image_path=None):
        assert len(prompt) > 0
        messages = [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": Image.new("RGB", (1,1), (0,0,0))},
                    ],
                },
        ] if image_path is None else [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, use_system=True)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)
        requests = [LLMInputs({ "prompt": text, "multi_modal_data": {"image": image_inputs} }),]
        sampling_params = SamplingParams(temperature=0, max_tokens=1, return_hidden_states=True)
        os.environ["IMAGE_GEN_MODE"] = "T2I" if image_path is None else "EDIT"
        outputs = self.model_vllm.generate(requests, sampling_params)
        prefill_hidden_states = outputs[0].prefill_hidden_states

        messages = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ] if image_path is None else [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                },
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)


        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        ).to(self.model_diffusion.device)

        for k in inputs.keys():
            if k in ["pixel_values", "pixel_values_videos", "audio_feats", "pixel_values_reference"]:
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        image = self.model_diffusion.generate(
            **inputs,
            image_gen_llm_hidden_states=prefill_hidden_states.unsqueeze(0),
            image_gen=True,
        )
        return image


if __name__ == '__main__':
    model = MingOmniImageGen(
        bl_mm = "/hetero_infer_new/serina.wzq/bailingv4_moe_lite/",
    )
    image_t2i = model.image_gen("生成一张图 美女在沙滩上奔跑")
    image_t2i.save("auto_t2i.jpg")

    image_edit = model.image_gen("给人物戴上眼镜", image_t2i)
    image_edit.save("auto_edit.jpg")