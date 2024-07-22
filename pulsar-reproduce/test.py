import torch
from stego_pipelines.longvideo import StegoFIFOVideoDiffusionPipeline
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StegoFIFOVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16")
pipe.to(device)

image_for_svd = [
    "input_sample.png",
    "bearded_man.jpg",
    "dog_run.jpg",
    "low_res_cat.jpg",
]
video_timesteps = 25
height, width = 512, 512

input_image_location = f"logging/images/for_svd/{image_for_svd[3]}"
image = utils.prepare_image(input_image_location, height, width)

pipeline_output = pipe.longify(
    stego_type=None,
    image = image,
    height=height,
    width=width,
    num_frames=16,
    num_inference_steps=video_timesteps,
    decode_chunk_size=3,
    video_length = 16,
    queue_length = 64,
    new_video_length = 100,
    num_partitions = 4,
)
