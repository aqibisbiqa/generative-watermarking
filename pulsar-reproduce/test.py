
import torch
import numpy as np
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image, ImageOps
from tqdm import tqdm
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise Exception("use gpu sir")

repo = "stabilityai/stable-video-diffusion-img2vid"
# pipe = StableVideoDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16, variant="fp16")
# pipe = pipe.to(device)


#@title Functions
# Function to rescale image and add padding if necessary
def prepare_image(image_path, target_width=1024, target_height=576):
    image = Image.open(image_path)

    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    target_aspect_ratio = target_width / target_height

    # Rescale the image to fit the target width or height while maintaining aspect ratio
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target aspect ratio
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Add padding to the image to match target dimensions
    padding_color = (0, 0, 0)  # Black padding
    image = ImageOps.pad(image, (target_width, target_height), color=padding_color)

    return image

# Modify the process_and_generate_video function to use prepare_image
def process_and_generate_video(image_path, repo, num_videos):
    for i in range(num_videos):
        try:
            # Load the model
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                repo,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            pipe.enable_model_cpu_offload()

            # Prepare the image
            image = prepare_image(image_path)
            # Set a seed for reproducibility (different for each video)
            generator = torch.manual_seed(42 + i)

            # Generate frames
            frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

            # Export to video
            # video_filename = os.path.join(folder_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_{i+1}.mp4")
            video_filename = "logging/images/for_svd/input_sample.mp4"
            export_to_video(frames, video_filename, fps=7)
            print(f"Video {i+1} generated successfully for {image_path}")
        except Exception as e:
            print(f"An error occurred while processing {image_path}: {e}")

image_path = "logging/images/for_svd/input_sample.png"

# Use a different seed for each video
# process_and_generate_video(image_path, repo, 1)

torch.cuda.empty_cache()