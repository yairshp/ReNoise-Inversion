import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ORIGINAL_PROMPTS_AND_OBJECTS_TO_ADD_PATH = (
    "data/original_prompts_and_objects_to_add.csv"
)
IMAGES_PATH = "data/mini_benchmark"
OUTPUT_PATH = "data/mini_benchmark.csv"


def get_data(data_path, images_path):
    data = pd.read_csv(data_path)
    bg_images_paths = [f"{images_path}/{p}.png" for p in data["original_prompt"]]
    original_prompts = data["original_prompt"].tolist()
    objects_to_add = data["object_to_add"].tolist()
    return bg_images_paths, original_prompts, objects_to_add


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


def preprocess_image(image_path, res=1024):
    image = Image.open(image_path)
    return image.convert("RGB").resize((res, res))


def main():
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf"
    ).to(DEVICE)
    llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    bg_images_paths, original_prompts, objects_to_add = get_data(
        ORIGINAL_PROMPTS_AND_OBJECTS_TO_ADD_PATH, IMAGES_PATH
    )
    bg_images = [preprocess_image(p) for p in bg_images_paths]

    query_format = "USER: <image>\nhere is the caption to the image: '{original_prompt}'. modify it so it includes {object_name}. ASSISTANT:"
    edit_prompts = get_edit_prompts(
        llava_processor,
        llava_model,
        query_format,
        bg_images,
        objects_to_add,
        original_prompts,
        DEVICE,
    )

    full_data = {
        "image_path": bg_images_paths,
        "original_prompt": original_prompts,
        "object_to_add": objects_to_add,
        "edit_prompt": edit_prompts,
    }
    full_data_df = pd.DataFrame(full_data)
    full_data_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
