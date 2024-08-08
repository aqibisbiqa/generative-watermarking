# import torch
# from stego_pipelines.longvideo import StegoFIFOVideoDiffusionPipeline, FIFOUNetSpatioTemporalConditionModel
# import utils

# device = "cuda" if torch.cuda.is_available() else "cpu"

# unet = FIFOUNetSpatioTemporalConditionModel.from_pretrained(
#     "stabilityai/stable-video-diffusion-img2vid",
#     subfolder="unet",
#     torch_dtype=torch.float16, 
#     variant="fp16",
# )
# pipe = StegoFIFOVideoDiffusionPipeline.from_pretrained(
#     "stabilityai/stable-video-diffusion-img2vid",
#     unet=unet,
#     torch_dtype=torch.float16, 
#     variant="fp16",
# )
# pipe.to(device)

# images_for_svd = [
#     "input_sample.png",
#     "bearded_man.jpg",
#     "dog_run.jpg",
#     "low_res_cat.jpg",
# ]
# video_timesteps = 25
# height, width = 512, 512

# input_image_location = f"svd_base_images/{images_for_svd[3]}"
# image = utils.prepare_image(input_image_location, height, width)

# pipeline_output = pipe.longify(
#     stego_type=None,
#     image = image,
#     height=height,
#     width=width,
#     num_frames=16,
#     num_inference_steps=video_timesteps,
#     decode_chunk_size=3,
#     video_length = 16,
#     queue_length = 64,
#     new_video_length = 100,
#     num_partitions = 4,
# )

import os
import shutil


model_type = "pixel"
src_path = f"../logging/{model_type}"
# dst_path = f"valid/stego/{model_type}"
for div_steps in os.listdir(src_path):
    if not div_steps.endswith(".txt"):
        src_temp = os.path.join(src_path, div_steps)
        # dst_temp = os.path.join(dst_path, div_steps)
        # dst_temp = dst_path
        # os.makedirs(dst_temp, exist_ok=True)
        for file in os.listdir(src_temp):
            if "cover" not in file and "encode" not in file: continue
            src = os.path.join(src_temp, file)
            num, _, _ = tuple(file.split("_"))
            num = int(num) - 1
            
            if "cover" in file:
                cover_or_stego = "cover"
                div = ""
            elif "encode" in file:
                cover_or_stego = "stego"
                div = f"/{div_steps}"

            else: continue
            
            train_or_valid = None
            if (num % 60) < 40:
                train_or_valid = "train"
                num = (num // 60) * 40 + (num%60)
            else:
                train_or_valid = "valid"
                num = (num // 60) * 20 + (num%60 - 40)
            num = str(num + 1)

            dst_dir = f"{train_or_valid}/{cover_or_stego}/{model_type}{div}"
            os.makedirs(dst_dir, exist_ok=True)
            dst = f"{dst_dir}/{num}.png"
            print(src, dst)
            # shutil.copy(src, dst)

        # f_path = os.path.join(path, file)
        # print(f_path)