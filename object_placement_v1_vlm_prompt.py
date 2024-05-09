import os
import sys
import pandas as pd
from PIL import Image
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from diffusers.utils import make_image_grid
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)
from PIL import Image

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig
from main import run as invert

from attention_maps_utils_by_timesteps import (
    get_attn_maps,
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
    preprocess,
    visualize_and_save_attn_map,
)


def get_config(args):
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_args()


def preprocess_image(image_path):
    image = Image.open(image_path)
    return image.convert("RGB").resize((512, 512))


def get_llava_model_and_processor(device):
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf"
    ).to(device)
    llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    return llava_model, llava_processor


def get_prompts(processor, model, images, device, object_names=None):
    if object_names is None:
        propmt = "USER: <image>\nWhat's in the image (answer in shortest way possible)? ASSISTANT:"
        prompts = [propmt for _ in range(len(images))]
    else:
        prompts = [
            f"USER: <image>\nWhat's in the image that a {ref} can be on (answer in shortest way possible)? ASSISTANT:"
            for ref in object_names
        ]
    answers = []
    for prompt, image in zip(prompts, images):
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        generate_ids = model.generate(**inputs, max_new_tokens=15)
        answer = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # the answer should be the string after the last colon
        answer = answer.split(":")[-1].strip()
        answers.append(answer)
    return answers


def get_edit_prompts(inversion_prompts, objects):
    edit_prompts = []
    for inversion_prompt, object in zip(inversion_prompts, objects):
        edit_prompts.append(f"a {inversion_prompt} and a {object}".lower())
    return edit_prompts


def invert_images(pipe_inversion, pipe_inference, images, prompts, run_config):
    inv_latents = []
    noises = []
    for image, prompt in zip(images, prompts):
        _, inv_latent, noise, _ = invert(
            image,
            prompt,
            run_config,
            pipe_inversion=pipe_inversion,
            pipe_inference=pipe_inference,
            do_reconstruction=False,
        )
        inv_latents.append(inv_latent)
        noises.append(noise)
    return inv_latents, noises


def get_run_config(model_type, scheduler_type):
    return RunConfig(
        model_type=model_type,
        scheduler_type=scheduler_type,
        noise_regularization_lambda_kl=0.08,
        noise_regularization_lambda_ac=40,
        num_inversion_steps=4,
        num_inference_steps=4,
    )


def main():
    config = get_config(sys.argv[1:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(config.data_path)
    positive_data = get_positive_data(data)

    llava_model, llava_processor = get_llava_model_and_processor()

    model_type = Model_Type.SDXL_Turbo
    scheduler_type = Scheduler_Type.EULER
    pipe_inversion, pipe_inference = get_pipes(
        model_type, scheduler_type, device=device
    )
    run_config = get_run_config(model_type, scheduler_type)

    for _, row in positive_data.iterrows():
        bg_images = preprocess_image(row["bg_img_path"])
        ref_images = preprocess_image(row["ref_img_path"])
        object_names = get_prompts(llava_processor, llava_model, ref_images, device)
        inversion_prompts = get_prompts(
            llava_processor, llava_model, bg_images, device, object_names
        )
        edit_prompts = get_edit_prompts(inversion_prompts, object_names)
        inv_latents, noises = invert_images(
            pipe_inversion, pipe_inference, bg_images, inversion_prompts, run_config
        )


def get_positive_data(data):
    return data[data["label"] == 1]


if __name__ == "__main__":
    main()
