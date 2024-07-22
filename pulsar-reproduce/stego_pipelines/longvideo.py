# from FIFO_Diffusion.opensora_fifo.sample.pipeline_videogen import VideoGenPipeline, VideoPipelineOutput
import numpy as np

# from opensora_fifo/sample/pipeline_videogen.py
import math
import html
import inspect
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union

import torch
import einops
from einops import rearrange
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, Transformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    BACKENDS_MAPPING,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from dataclasses import dataclass



# from opensora_fifo/sample/sample_fifo.py
import math
import os
import torch
import argparse
import torchvision

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

import os, sys
from tqdm import trange, tqdm

# sys.path.append(os.path.split(os.path.split(sys.path[0])[0])[0])
# sys.path.append(os.path.split(sys.path[0])[0])

import imageio
import copy

# from video
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# from ...image_processor import PipelineImageInput
# from ...models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
# from ...schedulers import EulerDiscreteScheduler
# from ...utils import BaseOutput, logging, replace_example_docstring
# from ...utils.torch_utils import is_compiled_module, randn_tensor
# from ...video_processor import VideoProcessor
# from ..pipeline_utils import DiffusionPipeline

from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers import StableVideoDiffusionPipeline

from rate_estimation import estimate_rate
from utils import mix_samples_using_payload


class StegoFIFOVideoDiffusionPipeline(StableVideoDiffusionPipeline):
    
    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__(
            vae,
            image_encoder,
            unet,
            scheduler,
            feature_extractor,
        )
    
    @torch.no_grad()
    def longify(
        self,
        stego_type: str,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 20,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        keys: tuple = (10, 11, 12),
        payload = None,
        version = "65x512x512",
        lookahead_denoising = True,
        video_length = 16,
        queue_length = 64,
        new_video_length = 100,
        num_partitions = 4,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        num_inference_steps = video_length * num_partitions
        queue_length = num_inference_steps
        
        # Instantiate base video model
        videogen_pipeline = self

        # Create directories for outputs (frames + latents)
        output_dir = "logging/longvideos"
        latents_dir = "logging/longvideos/latents"

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(latents_dir, exist_ok=True)
        
        print("The results will be saved in", output_dir)
        print("The latents will be saved in", latents_dir)
        
        # If latents already produced, skip FIFO process and just VAE decode
        fifo_proc = not os.path.exists(os.path.join(latents_dir, "fifo.pt"))
        if fifo_proc:
            num_vae = new_video_length // (video_length-1)    
            first_idx = video_length // 2 if lookahead_denoising else 0

            fifo_vae_video_frames = []
            for i in range(num_vae):
                target_latents = torch.cat(fifo_first_latents[first_idx+i*(video_length-1):first_idx+(i+1)*(video_length-1)+1], dim=2)
                video = videogen_pipeline.decode_latents(target_latents, new_video_length, decode_chunk_size)[0]

                if i == 0:
                    fifo_vae_video_frames.append(video)
                else:
                    fifo_vae_video_frames.append(video[1:])
            
            if num_vae > 0:
                fifo_vae_video_frames = torch.cat(fifo_vae_video_frames, dim=0)
                output_vae_path = os.path.join(output_dir, "fifo_vae.gif")
                fifo_vae_video_frames[0].save(output_vae_path, save_all=True, append_images=fifo_vae_video_frames[1:], optimize=False, duration=100, loop=0)
            return


        # Acquire latents from an origin video (skip if present)
        is_run_base = not os.path.exists(os.path.join(latents_dir, "video.pt"))
        if is_run_base:
            print("will run videogen_pipeline to get base video latents")
            latents = videogen_pipeline(
                image,
                height,
                width,
                video_length,
                # num_inference_steps,
                25,
                sigmas,
                min_guidance_scale,
                max_guidance_scale,
                fps,
                motion_bucket_id,
                noise_aug_strength,
                decode_chunk_size,
                num_videos_per_prompt,
                generator,
                latents,
                output_type="latent",
                return_dict=False,
            )

            torch.save(latents, os.path.join(latents_dir, "video.pt"))

            print("latents saved, now saving frames into GIF")

            # To see base video, need to VAE Decode + Postprocess 
            frames = videogen_pipeline.decode_latents(latents, num_frames, decode_chunk_size)
            frames = videogen_pipeline.video_processor.postprocess_video(video=frames, output_type="pil")[0]
            output_path = os.path.join(output_dir, "origin.gif")
            frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
        
        # FIFO scheduler utilizes more inference steps
        videogen_pipeline.scheduler.set_timesteps(num_inference_steps)
    
        # Prepare latents for FIFO diffusion
        latents = self.prepare_latents_FIFO(
            latents_dir, 
            videogen_pipeline.scheduler,
            lookahead_denoising,
            video_length,
            queue_length,
        ).to(torch.half)

        print(f"latents {latents.shape}")

        # To save individual FIFO frames
        save_frames = False
        if save_frames:
            fifo_dir = os.path.join(output_dir, "fifo_frames")
            os.makedirs(fifo_dir, exist_ok=True)
        
        fifo_video_frames = []
        fifo_first_latents = []

        # Parameter adjustment
        timesteps = videogen_pipeline.scheduler.timesteps
        indices = np.arange(num_inference_steps)
        # timesteps = torch.flip(timesteps, [0])
        
            # Lookahead denoising uses longer queue
        if lookahead_denoising:
            timesteps = torch.cat([torch.full((video_length//2,), timesteps[0]).to(timesteps.device), timesteps])
            indices = np.concatenate([np.full((video_length//2,), 0), indices])

            # Iterate for f + q - l
        num_iterations = new_video_length + num_inference_steps - video_length

        # Primary FIFO denoising loop
        for i in trange(num_iterations):
            # TODO: add multiprocessing support
            num_inference_steps_per_gpu = video_length

            # Lookahead denoising doubles number of partitions
            num_rank = 2 * num_partitions if lookahead_denoising else num_partitions

            # Diagonal denoising loop
            for rank in reversed(range(num_rank)):

                # Identify indices based on rank
                if lookahead_denoising:
                    start_idx = rank * (num_inference_steps_per_gpu // 2)
                else:
                    start_idx = rank * num_inference_steps_per_gpu
                midpoint_idx = start_idx + num_inference_steps_per_gpu // 2
                end_idx = start_idx + num_inference_steps_per_gpu

                print(f"indexes {start_idx} {midpoint_idx} {end_idx}")

                # Slice timesteps + latents based on rank
                t = timesteps[start_idx:end_idx]
                input_latents = latents[:,start_idx:end_idx].clone()

                # FIFO diffusion step
                output_latents, first_latent, first_frame = self.fifo_onestep(
                    stego_type,
                    image,
                    height=height,
                    width=width,
                    num_frames=video_length,
                    num_inference_steps=num_inference_steps,
                    sigmas=sigmas,
                    min_guidance_scale=min_guidance_scale,
                    max_guidance_scale=max_guidance_scale,
                    fps=fps,
                    motion_bucket_id=motion_bucket_id,
                    noise_aug_strength=noise_aug_strength,
                    decode_chunk_size=decode_chunk_size,
                    num_videos_per_prompt=num_videos_per_prompt,
                    generator=generator,
                    latents=input_latents,
                    video_length=video_length,
                    timesteps=t,
                    keys=keys,
                    payload=payload,
                )

                # Add denoised latents to queue
                if lookahead_denoising:
                    latents[:,midpoint_idx:end_idx] = output_latents[:,-(end_idx-midpoint_idx):]
                else:
                    latents[:,start_idx:end_idx] = output_latents
                del output_latents

            # Shift first {what goes here??} latents to back
            latents = self.shift_latents_FIFO(latents, videogen_pipeline.scheduler)

            fifo_first_latents.append(first_latent)

        # Save progress in case decoding errs
        torch.save(fifo_first_latents, os.path.join(latents_dir, "fifo.pt"))

        num_vae = new_video_length // (video_length-1)
        
        first_idx = video_length // 2 if lookahead_denoising else 0

        fifo_vae_video_frames = []
        for i in range(num_vae):
            target_latents = torch.cat(fifo_first_latents[first_idx+i*(video_length-1):first_idx+(i+1)*(video_length-1)+1], dim=2)
            video = videogen_pipeline.decode_latents(target_latents, new_video_length, decode_chunk_size)[0]

            if i == 0:
                fifo_vae_video_frames.append(video)
            else:
                fifo_vae_video_frames.append(video[1:])
        
        if num_vae > 0:
            fifo_vae_video_frames = torch.cat(fifo_vae_video_frames, dim=0)
            output_vae_path = os.path.join(output_dir, "fifo_vae.gif")
            fifo_vae_video_frames[0].save(output_vae_path, save_all=True, append_images=fifo_vae_video_frames[1:], optimize=False, duration=100, loop=0)
        
    @torch.no_grad()
    def fifo_onestep(
            self,
            stego_type: str,
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_frames: Optional[int] = None,
            num_inference_steps: int = 20,
            sigmas: Optional[List[float]] = None,
            min_guidance_scale: float = 1.0,
            max_guidance_scale: float = 3.0,
            fps: int = 7,
            # timesteps: Optional[torch.IntTensor] = None,
            # guidance_scale: float = 4.5,
            # eta: float = 0.0,
            motion_bucket_id: int = 127,
            noise_aug_strength: float = 0.02,
            decode_chunk_size: Optional[int] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            video_length: Optional[int] = None,
            rank: int = 0,
            timesteps: Optional[torch.IntTensor] = None,
            keys: tuple = (10, 11, 12),
            payload = None,
    ):
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        
        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 8. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # Custom stuff
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in keys])
        s_churn = 1.0

        # i want facts about the scheduler!
        sched = self.scheduler
        # print(f"sched sigmas are {sched.sigmas}")

        ####################################################
        ####################################################
        ####################################################
        ####################################################
        ####################################################
        ####################################################
        ####################################################
        ####################################################
        ####################################################
        
        latents = latents.to(device)
        # print(f"timesteps are {timesteps}")
        # print(f"input {latent_model_input.shape}")
        # print(f"slice is {latent_model_input[:, 0].shape}")

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        latent_model_input = torch.cat([self.scheduler.scale_model_input(latent_model_input[:, [i]], t) for i, t in enumerate(timesteps)], dim=1).to(device)
        # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        
        # Concatenate image_latents over channels dimension
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

        timesteps = timesteps.to(device)
        current_timestep = torch.mean(timesteps)


        # current_timestep = timesteps # torch.Size([f])
        # current_timestep = current_timestep[None].to(latent_model_input.device)
        # # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # current_timestep = current_timestep.expand(latent_model_input.shape[0], -1) # torch.Size([2, f])
        
        
        # latent_model_input should be [2, f, 8, 64, 64]
        # print(f"input {latent_model_input.shape}")

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            current_timestep,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # compute previous image: x_t -> x_t-1
        for i in range(video_length):
            latents[:,[i]] = self.scheduler.step(noise_pred[:,[i]], timesteps[i], latents[:,[i]], s_churn=0, generator=g_k_s).prev_sample
            self.scheduler._step_index = None

        
        first_latent = None
        first_frame = None
        if rank == 0:
            first_latent = latents[:,[0]]
            # if save_frames:
            #     first_frame = self.decode_latents(first_latent)[0]
            # else:
            #     first_frame = None

        return latents, first_latent, first_frame

    def prepare_latents_FIFO(
            self,
            latents_dir, 
            scheduler,
            lookahead_denoising,
            video_length,
            queue_length,
            generator=None,
        ):
        print("preparing latents for FIFO diffusion")
        timesteps = scheduler.timesteps
        
        latents_list = []
        video = torch.load(os.path.join(latents_dir, "video.pt")).to(timesteps.device)
        # note video is [b, f, c, h, w]
        print(f"video {video.shape}")

        if lookahead_denoising:
            for _ in range(video_length // 2):
                alpha = scheduler.alphas_cumprod[0]
                beta = 1 - alpha
                # latents = alpha**(0.5) * video[:,:,[0]] + beta**(0.5) * torch.randn(video[:,:,[0]].shape, generator=generator)
                latents = alpha**(0.5) * video[:,[0]] + beta**(0.5) * torch.randn(video[:,[0]].shape, generator=generator)
                latents_list.append(latents)

        for i in range(queue_length):
            alpha = scheduler.alphas_cumprod[i] # image -> noise
            beta = 1 - alpha
            frame_idx = max(0, i-(queue_length - video_length))
            # latents = (alpha)**(0.5) * video[:,:,[frame_idx]] + (1-alpha)**(0.5) * torch.randn(video[:,:,[frame_idx]].shape, generator=generator)
            latents = (alpha)**(0.5) * video[:,[frame_idx]] + (1-alpha)**(0.5) * torch.randn(video[:,[frame_idx]].shape, generator=generator)
            latents_list.append(latents)

        # latents = torch.cat(latents_list, dim=2)
        latents = torch.cat(latents_list, dim=1)
        return latents

    def shift_latents_FIFO(self, latents, scheduler, generator=None):
        # shift latents
        latents[:,:,:-1] = latents[:,:,1:].clone()

        # add new noise to the last frame
        latents[:,:,-1] = torch.randn_like(latents[:,:,-1]) * scheduler.init_noise_sigma

        return latents


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]