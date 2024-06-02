import os
import pprint
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt
from textwrap import wrap

from renoise_inversion.main import run as invert

# from attention_maps_utils_by_timesteps import (
#     get_attn_maps,
#     cross_attn_init,
#     register_cross_attention_hook,
#     set_layer_with_name_and_path,
#     preprocess,
#     visualize_and_save_attn_map,
# )
from attend_and_excite.utils import find_indices

MINI_BENCHMARK_OBJECT_NAMES = [
    "backpack",
    "flag",
    "picnic table",
    "dog",
    "bus",
    "picnic box",
    "plane",
    "bird",
    "bird",
    "car",
    "cat",
    "chair",
]


def get_llava_model_and_processor(device: str = "cuda"):
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf"
    ).to(device)
    llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    return llava_model, llava_processor


def get_generated_bg_images_and_prompts(images_dir: str):
    images_paths = os.listdir(images_dir)
    prompts = [p.split(".")[0] for p in images_paths]
    bg_images_paths = [f"{images_dir}/{p}" for p in images_paths]
    bg_images = [preprocess_image(p) for p in bg_images_paths]
    return bg_images, prompts


def get_object_names(images_dir):
    dir_basename = os.path.basename(images_dir)
    if dir_basename == "mini_benchmark":
        object_names = MINI_BENCHMARK_OBJECT_NAMES
    else:
        raise ValueError(f"Unknown directory name: {dir_basename}")
    return object_names


def preprocess_image(image_path, res=1024):
    image = Image.open(image_path)
    return image.convert("RGB").resize((res, res))


def get_edit_prompts(
    processor,
    model,
    query_format,
    images,
    object_names,
    original_prompts=None,
    device="cuda",
):
    edit_prompts = []
    queries = []
    if original_prompts is None:
        for object_name in object_names:
            query = query_format.format(object_name=object_names)
            queries.append(query)
    else:
        for original_prompt, object_name in zip(original_prompts, object_names):
            query = query_format.format(
                original_prompt=original_prompt, object_name=object_name
            )
            queries.append(query)
    for image, query in zip(images, queries):
        inputs = processor(text=query, images=image, return_tensors="pt").to(device)
        generate_ids = model.generate(**inputs, max_new_tokens=50)
        answer = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        answer = answer.split(":")[-1].strip()
        edit_prompts.append(answer)
    return edit_prompts


# def get_prompts(processor, model, images, object_names=None, device="cuda"):
#     if object_names is None:
#         propmt = "USER: <image>\nWhat's in the image (answer in shortest way possible)? ASSISTANT:"
#         prompts = [propmt for _ in range(len(images))]
#     else:
#         prompts = [
#             f"USER: <image>\nWhat's in the image that a {ref} can be on (answer in shortest way possible)? ASSISTANT:"
#             for ref in object_names
#         ]
#     answers = []
#     for prompt, image in zip(prompts, images):
#         inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
#         generate_ids = model.generate(**inputs, max_new_tokens=15)
#         answer = processor.batch_decode(
#             generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )[0]
#         # the answer should be the string after the last colon
#         answer = answer.split(":")[-1].strip()
#         answers.append(answer)
#     return answers


# def get_edit_prompts(inversion_prompts, objects):
#     edit_prompts = []
#     for inversion_prompt, object in zip(inversion_prompts, objects):
#         edit_prompts.append(f"a {inversion_prompt} and a {object}".lower())
#     return edit_prompts


def invert_images(config, pipe_inversion, pipe_inference, images, prompts):
    inv_latents = []
    noises = []
    for image, prompt in zip(images, prompts):
        _, inv_latent, noise, _ = invert(
            image,
            prompt,
            config,
            pipe_inversion=pipe_inversion,
            pipe_inference=pipe_inference,
            do_reconstruction=False,
        )
        inv_latents.append(inv_latent)
        noises.append(noise)
    return inv_latents, noises


# def set_pipe_for_cross_attn_logging(pipe_inference):
#     cross_attn_init()
#     pipe_inference.unet = set_layer_with_name_and_path(pipe_inference.unet)
#     pipe_inference.unet = register_cross_attention_hook(pipe_inference.unet)
#     return pipe_inference


# def get_attn_map(pipe_inference, edit_prompt):
#     attn_maps = get_attn_maps()
#     attn_map = preprocess(attn_maps[-1], 512, 512)
#     attn_map_img = visualize_and_save_attn_map(
#         attn_map, pipe_inference.tokenizer, edit_prompt, edit_prompt.split()[-1].lower()
#     )
#     return attn_map_img
# def get_attn_map(edit_prompt, tokenizer, timesteps, indices):
#     attn_maps = get_attn_maps()
#     attn_maps_images = []
#     for t in timesteps:
#         attn_map = preprocess(attn_maps[t], 512, 512)
#         # attn_map = preprocess(attn_maps[-1], 512, 512)
#         attn_map_img = visualize_and_save_attn_map(
#             attn_map,
#             tokenizer,
#             edit_prompt,
#             indices=indices,
#         )
#         attn_maps_images.append(attn_map_img)
#     return attn_maps_images


# def reset_attn_maps():
#     attn_maps = get_attn_maps()
#     attn_maps.clear()


def get_edit_images_and_attn_maps(
    config,
    pipe_inference,
    inv_latents,
    edit_prompts,
    objects_to_add,
    inference_guidance_scale,
    is_aae=False,
    aae_max_iter_to_alter=None,
    aae_thresholds=None,
    aae_scale_factor=None,
):
    edit_images = []
    last_timestep_attn_maps = []
    for inv_latent, edit_prompt, object_name in zip(
        inv_latents, edit_prompts, objects_to_add
    ):
        pipe_inference.cfg = config
        if is_aae:
            tokens_indices = find_indices(edit_prompt, [object_name], pipe_inference)
            edit_image, attn_map = pipe_inference(
                prompt=edit_prompt,
                num_inference_steps=config.num_inference_steps,
                image=inv_latent,
                token_indices=tokens_indices,
                strength=config.inversion_max_step,
                denoising_start=1.0 - config.inversion_max_step,
                guidance_scale=inference_guidance_scale,
                disable_aae=False,
                max_iter_to_alter=aae_max_iter_to_alter,
                scale_factor=aae_scale_factor,
                thresholds=aae_thresholds,
            )
            edit_images.append(edit_image.images[0])
            last_timestep_attn_maps.append(attn_map)
        else:
            edit_image = pipe_inference(
                prompt=edit_prompt,
                num_inference_steps=config.num_inference_steps,
                image=inv_latent,
                strength=config.inversion_max_step,
                denoising_start=1.0 - config.inversion_max_step,
                guidance_scale=inference_guidance_scale,
            )
            edit_images.append(edit_image.images[0])

    return edit_images, last_timestep_attn_maps


# def get_indices_to_alter(stable, prompt: str):
#     token_idx_to_word = {
#         idx: stable.tokenizer.decode(t)
#         for idx, t in enumerate(stable.tokenizer(prompt)["input_ids"])
#         if 0 < idx < len(stable.tokenizer(prompt)["input_ids"]) - 1
#     }
#     pprint.pprint(token_idx_to_word)
#     token_indices = input(
#         "Please enter the a comma-separated list indices of the tokens you wish to "
#         "alter (e.g., 2,5): "
#     )
#     token_indices = [int(i) for i in token_indices.split(",")]
#     print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
#     return token_indices


def plot_images_and_prompts(
    original_images,
    original_prompts,
    edit_images,
    edit_prompts,
    object_names,
    display=True,
):
    # Create a figure with the desired number of rows and columns
    num_images = len(original_images)

    rows = num_images
    cols = 2  # Two columns for original and edit image pairs
    fig, axs = plt.subplots(rows, cols, figsize=(10, rows * 4), constrained_layout=True)

    # Iterate over the lists and plot each image pair in a row
    for i in range(num_images):

        # Load and display the original image
        axs[i, 0].imshow(original_images[i])
        axs[i, 0].set_title("\n".join(wrap(original_prompts[i], 40)), fontsize=10)
        axs[i, 0].axis("off")

        # Load and display the edit image
        wrapped_prompt = "\n".join(wrap(edit_prompts[i], 40))
        bold_object_name = r"$\bf{" + object_names[i].replace(" ", "\ ") + "}$"
        axs[i, 1].imshow(edit_images[i])
        axs[i, 1].set_title(
            wrapped_prompt.replace(object_names[i], bold_object_name), fontsize=10
        )
        # axs[i, 1].set_title('\n'.join(wrap(edit_prompts[i], 30)), fontsize=10)
        axs[i, 1].axis("off")

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.4)

    # Display the plot
    if display:
        plt.show()
    else:
        return fig
