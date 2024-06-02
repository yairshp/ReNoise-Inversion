import torch
import wandb
import numpy as np
import pandas as pd
from datetime import datetime

from torchmetrics.functional.multimodal import clip_score
from functools import partial

import object_placement_utils as utils
from renoise_inversion.eunms import Model_Type, Scheduler_Type
from renoise_inversion.utils.enums_utils import get_pipes
from renoise_inversion.config import RunConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = Model_Type.SDXL
SCHEDULER_TYPE = Scheduler_Type.DDIM
DATA_PATH = "data/mini_benchmark.csv"

HYPERPARAMS = {
    "is_aae": False,
    "seed": 7865,
    "num_inference_steps": 50,
    "num_inversion_steps": 50,
    "num_renoise_steps": 1,
    "noise_regularization_lambda_ac": 0,
    "noise_regularization_lambda_kl": 0,
    "perform_noise_correction": False,
    "inference_guidance_scale": 10,
    "aae_max_iter_to_alter": 25,
    "aae_thresolds": {0: 0.05, 10: 0.5, 20: 0.8},
    "aae_scale_factor": 10,
}

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def get_data(data_path):
    data = pd.read_csv(data_path)
    bg_images_paths = data["image_path"].tolist()
    original_prompts = data["original_prompt"].tolist()
    objects_to_add = data["object_to_add"].tolist()
    edit_prompts = data["edit_prompt"].tolist()
    return bg_images_paths, original_prompts, objects_to_add, edit_prompts


def get_inversion_run_config(params):
    return RunConfig(
        model_type=MODEL_TYPE,
        scheduler_type=SCHEDULER_TYPE,
        seed=params["seed"],
        num_inference_steps=params["num_inference_steps"],
        num_inversion_steps=params["num_inversion_steps"],
        num_renoise_steps=params["num_renoise_steps"],
        noise_regularization_lambda_ac=params["noise_regularization_lambda_ac"],
        noise_regularization_lambda_kl=params["noise_regularization_lambda_kl"],
        perform_noise_correction=params["perform_noise_correction"],
    )


def get_mses(bg_images, edit_images):
    mses = []
    for bg_image, edit_image in zip(bg_images, edit_images):
        bg_image_arr = torch.from_numpy(np.array(bg_image).astype("uint8") / 255)
        edit_image_arr = torch.from_numpy(np.array(edit_image).astype("uint8") / 255)
        mse = torch.nn.functional.mse_loss(bg_image_arr, edit_image_arr)
        mses.append(mse)
    mses = torch.stack(mses)
    return mses


def get_clip_scores(edit_prompts, edit_images):
    clip_scores = []
    for edit_image, edit_prompt in zip(edit_images, edit_prompts):
        edit_image_arr = np.array(edit_image).astype("uint8")
        edit_image_and_prompt_clip_score = clip_score_fn(
            torch.from_numpy(edit_image_arr).permute(2, 0, 1), edit_prompt
        ).detach()
        clip_scores.append(edit_image_and_prompt_clip_score)
    clip_scores = torch.stack(clip_scores)
    return clip_scores


def main():
    print(HYPERPARAMS)
    bg_images_paths, original_promtps, objects_to_add, edit_prompts = get_data(
        DATA_PATH
    )
    bg_images = [utils.preprocess_image(p) for p in bg_images_paths]
    inversion_run_config = get_inversion_run_config(HYPERPARAMS)
    pipe_inversion, pipe_inference = get_pipes(
        MODEL_TYPE, SCHEDULER_TYPE, device=DEVICE, is_aae=HYPERPARAMS["is_aae"]
    )
    inv_latents, noises = utils.invert_images(
        config=inversion_run_config,
        pipe_inversion=pipe_inversion,
        pipe_inference=pipe_inference,
        images=bg_images,
        prompts=original_promtps,
    )
    edit_images, attn_maps = utils.get_edit_images_and_attn_maps(
        config=inversion_run_config,
        pipe_inference=pipe_inference,
        inference_guidance_scale=HYPERPARAMS["inference_guidance_scale"],
        inv_latents=inv_latents,
        edit_prompts=edit_prompts,
        objects_to_add=objects_to_add,
        is_aae=HYPERPARAMS["is_aae"],
        aae_max_iter_to_alter=HYPERPARAMS["aae_max_iter_to_alter"],
        aae_thresholds=HYPERPARAMS["aae_thresolds"],
        aae_scale_factor=HYPERPARAMS["aae_scale_factor"],
    )
    wandb.init(
        entity="yairshp",
        project="2d_object_placement",
        job_type="inference",
        config=HYPERPARAMS,
        name=f"inversion_aae_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
    )
    results = utils.plot_images_and_prompts(
        original_images=bg_images,
        original_prompts=original_promtps,
        edit_images=edit_images,
        edit_prompts=edit_prompts,
        object_names=objects_to_add,
        display=False,
    )
    clip_scores = get_clip_scores(edit_prompts, edit_images)
    mses = get_mses(bg_images, edit_images)

    wandb.log(
        {
            "results": results,
            "clip_score": clip_scores.mean().item(),
            "mse": mses.mean().item(),
        }
    )

    for i, attn_map in enumerate(attn_maps):
        attn_map.save(f"example_images/attn_maps/{i}.png")


if __name__ == "__main__":
    main()
