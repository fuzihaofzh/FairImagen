import json
import random
import shutil
import torch

import hyperparse
import fire
import os
import tqdm
from PIL import Image
from eval_tool import *
from fairpca_processor import *
from sdpipline import UserStableDiffusion3Pipeline
from pipeline_factory import create_pipeline, get_supported_pipelines, get_pipeline_info
from pipeline_adapters import create_adapted_pipeline

def stitch_images(images, num_columns):
    """
    Stitch multiple images into a single large image and return it.

    Args:
    images (list): List of images.
    num_columns (int): Number of images to display per row.

    Returns:
    Image: The stitched large image object.
    """
    # Automatically calculate the number of rows
    num_rows = (len(images) + num_columns - 1) // num_columns

    # Assume all images have the same size, get the size of a single image
    image_width, image_height = images[0].size

    # Create a blank large image
    big_image = Image.new(
        'RGB', 
        (num_columns * image_width, num_rows * image_height), 
        (255, 255, 255)  # Set the background color to white
    )

    # Paste each small image into the large image one by one
    for index, image in enumerate(images):
        row = index // num_columns
        col = index % num_columns
        big_image.paste(image, (col * image_width, row * image_height))

    # Return the stitched large image object
    return big_image




    

def demo(pipe):
    images = pipe(
        "a photo of the face of a firefighter",
        num_images_per_prompt=20,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images


def extract_features(pipe):
    directory = pipe.exp_dir
    #os.system(f"rm -rf '{directory}'")
    for protect in pipe.usermode['protect']:
        prompts = json.load(open(f"data/{protect}.json"))
        prompts = pipe.processor.process_input(prompts, pipe.usermode, protect)
        for cat in prompts:
            print(protect, ":", cat)
            for pmpt in tqdm.tqdm(prompts[cat]):
                images = pipe(
                    pmpt,
                    num_images_per_prompt=1,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    protect = protect,
                    cat = cat
                )

def generate(pipe):
    prompt = "a photo of the face of a firefighter"
    if "prompt" in pipe.usermode:
        prompt = f"a photo of the face of {pipe.usermode['prompt'].replace('_', ' ')}"
    # prompt = "Generate a photo of a face of a firefighter"
    # prompt = "a photo of a face of a firefighter"
    num_images_per_prompt = 16
    prompt_ori = prompt
    prompt = pipe.processor.modify_prompt(prompt, pipe.usermode, num_images_per_prompt)
    print(f"\n\nPrompt: {prompt}\n\n")
    images = pipe(
        prompt,
        #"a photo of the face of a lumberjack with fact unobstructed.",
        #"A kora player seated by a tree, with the face close-up as playing, with a backdrop of an village and people gathered to listen.",
        #"A serene tea master in ceremonial attire looking at the camera, pouring tea from an elegant pot, with soft floral patterns subtly visible on their clothing.",
        num_images_per_prompt=num_images_per_prompt if type(prompt) is str else 1,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images
    directory = os.path.join(pipe.exp_dir)#, "output_images")
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    #for i, image in enumerate(images):
    #    image.save(os.path.join(directory, f"{i}.png"))

    # Call the stitching function to combine the generated images into a single large image
    big_image = stitch_images(images, num_columns=4)

    # Output path for the stitched image
    big_image_path = os.path.join(directory, f"output_{pipe.extramode_str}.jpeg")

    # Save the stitched image with compression
    big_image.save(big_image_path, optimize=True, quality=80)
    print(img_evaluate(big_image_path, pipe.usermode, prompt_ori, images[0].size))

def run(pipe, usermode):
    data = json.load(open(f"data/{usermode['data']}.json"))
    # Use encoder-specific feature filename
    feature_filename = pipe.processor.get_feature_filename(usermode)
    feature_path = os.path.join(pipe.exp_dir, feature_filename)
    if not os.path.exists(feature_path):
        print("Extracting features")
        pipe.usermode["extract"] = True
        extract_features(pipe)
        del pipe.usermode["extract"]
    scores = []
    directory = os.path.join(pipe.exp_dir, f"{pipe.extramode_str}")
    # Adjust batch size based on pipeline
    if usermode.get("pipe", "").startswith("sdxl"):
        num_images_per_prompt = 4  # Smaller batch for SDXL due to memory
    else:
        num_images_per_prompt = 12 #24
    height = 256
    width = 256
    for prompt in data:
        imgprompt = pipe.processor.modify_prompt(prompt, pipe.usermode, num_images_per_prompt)
        #if len(prompt.split(" ")) <= 2:
        #    imgprompt = f"Generate a photo of {prompt}"
        print(imgprompt)
        # Use pipeline's default dimensions
        height = None
        width = None
        num_inference_steps = usermode.get("istep", 10)#28
        #if usermode["proc"] == "t2i":
        #    height = 256
        #    width = 256
        #    num_inference_steps = 14
        images = pipe(
            imgprompt,
            num_images_per_prompt=num_images_per_prompt if type(imgprompt) is str else 1,
            negative_prompt="",
            num_inference_steps=num_inference_steps,
            guidance_scale=7.0,
            height = height,
            width = width,
        ).images
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Call the stitching function to combine the generated images into a single large image
        big_image = stitch_images(images, num_columns=4)

        # Output path for the stitched image
        big_image_path = os.path.join(directory, f"{prompt}.jpeg")

        # Save the stitched image with compression
        big_image.save(big_image_path, optimize=True, quality=80)

        size = images[0].size
        try:
            score = img_evaluate(big_image_path, usermode, prompt, size)
            score = {s : score[s] for s in score if s != 'patches'}
            scores.append(score)
        except Exception as e:
            print(e)
    json.dump(scores, open(os.path.join(pipe.exp_dir, f"{pipe.extramode_str}", "full_scores.json"), "w"))
    avg_scores = {k: sum([s[k] if type(s[k]) is not str else 0 for s in scores]) / len(scores) for k in scores[0] if type(scores[0][k]) in [float, int]}
    json.dump(avg_scores, open(os.path.join(pipe.exp_dir, f"{pipe.extramode_str}", "scores.json"), "w"))
    print(avg_scores)



    
def main(usermode_str = "data=dev,protect=[gender]", extramode_str = ""):
    print(usermode_str, extramode_str)
    usermode = hyperparse.parse_string(usermode_str)
    extramode = hyperparse.parse_string(extramode_str)
    usermode.update(extramode)
    pipename = usermode.get("pipe", "sd3.0")
    
    # Display supported pipelines if requested
    if pipename == "list":
        print("Supported pipelines:")
        for p in get_supported_pipelines():
            info = get_pipeline_info(p)
            print(f"  {p}: {info['type']} ({info['size']}, {info['speed']})")
        return
    
    # Create pipeline using factory
    try:
        pipe = create_pipeline(pipename)
        print(f"✅ Created pipeline: {pipename}")
        pipeline_info = get_pipeline_info(pipename)
        print(f"   Type: {pipeline_info['type']}, Size: {pipeline_info['size']}, Speed: {pipeline_info['speed']}")
    except Exception as e:
        print(f"❌ Error creating pipeline '{pipename}': {e}")
        print("Supported pipelines:", get_supported_pipelines())
        return
    

    pipe.exp_dir = os.path.join("output/exps", usermode_str)
    pipe.extramode_str = extramode_str
    pipe.usermode = usermode
    
    # Check if pipeline needs adaptation for processor support
    needs_adaptation = False  # Wurstchen now has its own adapter
    
    pipe = pipe.to("cuda")
    if "proc" in usermode:
        if usermode["proc"] == "fpca":
            pipe.processor = FairPCAProcessor()
        elif usermode["proc"] == "base":
            pipe.processor = BaseProcessor()
    else:
        pipe.processor = FairPCAProcessor()
    
    # Apply generic adapter for pipelines that need it
    if needs_adaptation:
        pipe = create_adapted_pipeline(pipe, pipe.processor, usermode, pipe.exp_dir)
    
    # set random seed for reproducibility
    if "seed" in usermode:
        torch.manual_seed(usermode["seed"])
        torch.cuda.manual_seed(usermode["seed"])
        torch.cuda.manual_seed_all(usermode["seed"])
        random.seed(usermode["seed"])
        np.random.seed(usermode["seed"])
    if "extract" in usermode:
        extract_features(pipe)
    elif "data" in usermode:
        run(pipe, usermode)
    else:
        generate(pipe)

if __name__ == "__main__":
    fire.Fire(main)