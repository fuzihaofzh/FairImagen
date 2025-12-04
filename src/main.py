"""Main entry point for FairImagen - fairness-aware text-to-image generation.

This module provides the main pipeline for:
- Extracting demographic features from prompts
- Generating images with fairness-aware processing
- Running evaluation benchmarks on generated images
"""

import json
import random
from pathlib import Path

import fire
import hyperparse
import numpy as np
import torch
import tqdm
from PIL import Image

from base_processor import BaseProcessor
from eval_tool import img_evaluate
from fairpca_processor import FairPCAProcessor
from pipeline_adapters import create_adapted_pipeline
from pipeline_factory import create_pipeline, get_pipeline_info, get_supported_pipelines


def stitch_images(images, num_columns):
    """Stitch multiple images into a single large image and return it.

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
        "RGB",
        (num_columns * image_width, num_rows * image_height),
        (255, 255, 255),  # Set the background color to white
    )

    # Paste each small image into the large image one by one
    for index, image in enumerate(images):
        row = index // num_columns
        col = index % num_columns
        big_image.paste(image, (col * image_width, row * image_height))

    # Return the stitched large image object
    return big_image


def demo(pipe) -> None:
    """Run a demo generation of firefighter faces."""
    _result = pipe(
        "a photo of the face of a firefighter",
        num_images_per_prompt=20,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images


def extract_features(pipe) -> None:
    """Extract fairness features from prompts for different protected attributes."""
    for protect in pipe.usermode["protect"]:
        with Path(f"data/{protect}.json").open() as f:
            prompts = json.load(f)
        prompts = pipe.processor.process_input(prompts, pipe.usermode, protect)
        for cat in prompts:
            for pmpt in tqdm.tqdm(prompts[cat]):
                _result = pipe(
                    pmpt,
                    num_images_per_prompt=1,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    protect=protect,
                    cat=cat,
                )


def generate(pipe) -> None:
    """Generate images for a single prompt and evaluate fairness metrics."""
    prompt = "a photo of the face of a firefighter"
    if "prompt" in pipe.usermode:
        prompt = f"a photo of the face of {pipe.usermode['prompt'].replace('_', ' ')}"
    num_images_per_prompt = 16
    prompt = pipe.processor.modify_prompt(prompt, pipe.usermode, num_images_per_prompt)
    images = pipe(
        prompt,
        num_images_per_prompt=num_images_per_prompt if isinstance(prompt, str) else 1,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images
    directory = Path(pipe.exp_dir)
    directory.mkdir(parents=True, exist_ok=True)

    # Call the stitching function to combine the generated images into a single large image
    big_image = stitch_images(images, num_columns=4)

    # Output path for the stitched image
    big_image_path = directory / f"output_{pipe.extramode_str}.jpeg"

    # Save the stitched image with compression
    big_image.save(big_image_path, optimize=True, quality=80)


def run(pipe, usermode) -> None:
    """Run full evaluation pipeline on dataset with fairness metrics."""
    with Path(f"data/{usermode['data']}.json").open() as f:
        data = json.load(f)
    # Use encoder-specific feature filename
    feature_filename = pipe.processor.get_feature_filename(usermode)
    feature_path = Path(pipe.exp_dir) / feature_filename
    if not feature_path.exists():
        pipe.usermode["extract"] = True
        extract_features(pipe)
        del pipe.usermode["extract"]
    scores = []
    directory = Path(pipe.exp_dir) / f"{pipe.extramode_str}"
    # Adjust batch size based on pipeline
    num_images_per_prompt = 4 if usermode.get("pipe", "").startswith("sdxl") else 12
    height = 256
    width = 256
    for prompt in data:
        imgprompt = pipe.processor.modify_prompt(
            prompt,
            pipe.usermode,
            num_images_per_prompt,
        )
        # Use pipeline's default dimensions
        height = None
        width = None
        num_inference_steps = usermode.get("istep", 10)
        images = pipe(
            imgprompt,
            num_images_per_prompt=(
                num_images_per_prompt if isinstance(imgprompt, str) else 1
            ),
            negative_prompt="",
            num_inference_steps=num_inference_steps,
            guidance_scale=7.0,
            height=height,
            width=width,
        ).images
        directory.mkdir(parents=True, exist_ok=True)

        # Call the stitching function to combine the generated images into a single large image
        big_image = stitch_images(images, num_columns=4)

        # Output path for the stitched image
        big_image_path = directory / f"{prompt}.jpeg"

        # Save the stitched image with compression
        big_image.save(big_image_path, optimize=True, quality=80)

        size = images[0].size
        try:
            score = img_evaluate(big_image_path, usermode, prompt, size)
            score = {s: score[s] for s in score if s != "patches"}
            scores.append(score)
        except (RuntimeError, ValueError, KeyError):
            pass
    with (Path(pipe.exp_dir) / f"{pipe.extramode_str}" / "full_scores.json").open(
        "w",
    ) as f:
        json.dump(scores, f)
    avg_scores = {
        k: sum([s[k] if type(s[k]) is not str else 0 for s in scores]) / len(scores)
        for k in scores[0]
        if type(scores[0][k]) in [float, int]
    }
    with (Path(pipe.exp_dir) / f"{pipe.extramode_str}" / "scores.json").open("w") as f:
        json.dump(avg_scores, f)


def main(usermode_str="data=dev,protect=[gender]", extramode_str="") -> None:
    """Main entry point for FairImagen pipeline.

    Args:
        usermode_str: Configuration string (e.g., "data=dev,protect=[gender]")
        extramode_str: Additional configuration options

    """
    usermode = hyperparse.parse_string(usermode_str)
    extramode = hyperparse.parse_string(extramode_str)
    usermode.update(extramode)
    pipename = usermode.get("pipe", "sd3.0")

    # Display supported pipelines if requested
    if pipename == "list":
        for p in get_supported_pipelines():
            get_pipeline_info(p)
        return

    # Create pipeline using factory
    try:
        pipe = create_pipeline(pipename)
        get_pipeline_info(pipename)
    except (ValueError, ImportError, RuntimeError):
        return

    pipe.exp_dir = Path("output/exps") / usermode_str
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
        # Prefer Generator API over legacy seed
        _rng = np.random.default_rng(usermode["seed"])  # noqa: F841
    if "extract" in usermode:
        extract_features(pipe)
    elif "data" in usermode:
        run(pipe, usermode)
    else:
        generate(pipe)


if __name__ == "__main__":
    fire.Fire(main)
