import os
import sys
import contextlib
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser

import object_placement_utils as utils
from renoise_inversion.eunms import Model_Type, Scheduler_Type
from renoise_inversion.utils.enums_utils import get_pipes
from renoise_inversion.config import RunConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args(args):
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_type",
        default=Model_Type.SDXL_Turbo,
    )
    parser.add_argument(
        "--scheduler_type",
        default=Scheduler_Type.EULER,
    )
    parser.add_argument("--is_aae", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=7865)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--num_inversion_steps", type=int, default=4)
    parser.add_argument("--num_renoise_steps", type=int, default=9)
    parser.add_argument("--noise_regularization_lambda_ac", type=float, default=40)
    parser.add_argument("--noise_regularization_lambda_kl", type=float, default=0.1)
    parser.add_argument("--perform_noise_correction", type=bool, default=True)
    parser.add_argument("--inference_guidance_scale", type=float, default=0.0)
    return parser.parse_args(args)


def get_data(data_path):
    # exptected columns: ["bg_image_path", "original_prompt", "object_to_add", "edit_prompt", "filename"]
    data = pd.read_csv(data_path)
    return data


def get_inversion_run_config(params):
    return RunConfig(
        model_type=params["model_type"],
        scheduler_type=params["scheduler_type"],
        seed=params["seed"],
        num_inference_steps=params["num_inference_steps"],
        num_inversion_steps=params["num_inversion_steps"],
        num_renoise_steps=params["num_renoise_steps"],
        noise_regularization_lambda_ac=params["noise_regularization_lambda_ac"],
        noise_regularization_lambda_kl=params["noise_regularization_lambda_kl"],
        perform_noise_correction=params["perform_noise_correction"],
    )


def main():
    args = get_args(sys.argv[1:])
    print(args)
    inversion_run_config = get_inversion_run_config(vars(args))
    pipe_inversion, pipe_inference = get_pipes(
        args.model_type, args.scheduler_type, DEVICE
    )
    data = get_data(args.data_path)
    bg_images_paths = data["bg_image_path"].tolist()
    bg_images = [utils.preprocess_image(img, 512) for img in bg_images_paths]
    original_prompts = data["original_prompt"].tolist()
    edit_prompts = data["edit_prompt"].tolist()
    filenames = data["filename"].tolist()
    inv_latents, noises = utils.invert_images(
        config=inversion_run_config,
        pipe_inversion=pipe_inversion,
        pipe_inference=pipe_inference,
        images=bg_images,
        prompts=original_prompts,
    )
    for inv_latent, noise, original_prompt, edit_prompt, filename in zip(
        inv_latents, noises, original_prompts, edit_prompts, filenames
    ):
        if args.perform_noise_correction:
            pipe_inference.scheduler.set_noise_list(noise)
        edit_image = pipe_inference(
            prompt=edit_prompt,
            image=inv_latent,
            negative_prompt=original_prompt,
            num_inference_steps=inversion_run_config.num_inference_steps,
            strength=inversion_run_config.inversion_max_step,
            denoising_start=1.0 - inversion_run_config.inversion_max_step,
            guidance_scale=args.inference_guidance_scale,
        ).images[0]
        edit_image.save(os.path.join(args.output_dir, f"{filename}.png"))


if __name__ == "__main__":
    main()
