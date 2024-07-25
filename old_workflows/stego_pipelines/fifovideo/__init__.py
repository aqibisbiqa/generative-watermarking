from FIFO_Diffusion.opensora_fifo.sample.pipeline_videogen import VideoGenPipeline, VideoPipelineOutput
from FIFO_Diffusion.opensora_fifo.sample.sample_fifo import shift_latents, prepare_latents
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

sys.path.append(os.path.split(os.path.split(sys.path[0])[0])[0])
from FIFO_Diffusion.opensora_fifo.models.diffusion.latte.modeling_latte import LatteT2V
from FIFO_Diffusion.opensora.models.ae import ae_stride_config, getae, getae_wrapper
from FIFO_Diffusion.opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from FIFO_Diffusion.opensora.models.text_encoder import get_text_enc
from FIFO_Diffusion.opensora.utils.utils import save_video_grid

sys.path.append(os.path.split(sys.path[0])[0])

import imageio
import copy


class StegoFIFOVideoDiffusionPipeline(VideoGenPipeline):
    
    def __init__(
            self,
            tokenizer: T5Tokenizer,
            text_encoder: T5EncoderModel,
            vae: AutoencoderKL,
            transformer: Transformer2DModel,
            scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__(
            tokenizer,
            text_encoder,
            vae,
            transformer,
            scheduler,
        )
    
    def __call__(
            self,
            stego_type: str,
            # image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
            prompt: Union[str, List[str]] = None,
            negative_prompt: str = "",
            num_inference_steps: int = 20,
            timesteps: List[int] = None,
            guidance_scale: float = 4.5,
            num_images_per_prompt: Optional[int] = 1,
            num_frames: Optional[int] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            clean_caption: bool = True,
            mask_feature: bool = True,
            enable_temporal_attentions: bool = True,
            save_latents: bool = False,
            latents_dir: str = None,
    ) -> Union[VideoPipelineOutput, Tuple]:
        
        args = {
            "model_path": "LanguageBind/Open-Sora-Plan-v1.1.0",
            "text_encoder_name": "DeepFloyd/t5-v1_1-xxl",
            "text_prompt": "FIFO_Diffusion/prompts/test_prompts_opensora.txt",
            "ae": "CausalVAEModel_4x8x8",
            "version": "65x512x512",
            "sample_method": "DDPM",
            "fps": 24,
            "guidance_scale": 7.5,
            "enable_tiling": True,
            "cache_dir": "opensoraplan_models/",
            "num_frames": 65,
            "video_length": 17,
            "new_video_length": 100,
            "num_partitions": 8,
        }

        # torch.manual_seed(args["seed"])
        torch.set_grad_enabled(False)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        vae = getae_wrapper(args["ae"])(args["model_path"], subfolder="vae", cache_dir=args["cache_dir"]).to(device, dtype=torch.float16)
        if args["enable_tiling"]:
            vae.vae.enable_tiling()
            vae.vae.tile_overlap_factor = args["tile_overlap_factor"]
        vae.vae_scale_factor = ae_stride_config[args["ae"]]
        # Load model:
        transformer_model = LatteT2V.from_pretrained(args["model_path"], subfolder=args["version"], cache_dir=args["cache_dir"], torch_dtype=torch.float16).to(device)
        transformer_model.force_images = args["force_images"]
        tokenizer = T5Tokenizer.from_pretrained(args["text_encoder_name"], cache_dir=args["cache_dir"])
        text_encoder = T5EncoderModel.from_pretrained(args["text_encoder_name"], cache_dir=args["cache_dir"], torch_dtype=torch.float16).to(device)

        # video_length, image_size = transformer_model.config.video_length, int(args["version"].split('x')[1])
        # latent_size = (image_size // ae_stride_config[args["ae"]][1], image_size // ae_stride_config[args["ae"]][2])
        # vae.latent_size = latent_size
        if args["force_images"]:
            video_length = 1
            ext = 'jpg'
        else:
            ext = 'mp4'

        # set eval mode
        transformer_model.eval()
        vae.eval()
        text_encoder.eval()

        schedulers = None
        if args["sample_method"] == 'DDIM':  #########
            scheduler = DDIMScheduler()
        elif args["sample_method"] == 'EulerDiscrete':
            scheduler = EulerDiscreteScheduler()
        elif args["sample_method"] == 'DDPM':  #############
            scheduler = DDPMScheduler()
        elif args["sample_method"] == 'DPMSolverMultistep':
            scheduler = DPMSolverMultistepScheduler()
        elif args["sample_method"] == 'DPMSolverSinglestep':
            scheduler = DPMSolverSinglestepScheduler()
        elif args["sample_method"] == 'PNDM':
            scheduler = PNDMScheduler()
            schedulers = [PNDMScheduler() for _ in range(args["video_length"])]
            for s in schedulers:
                s.set_timesteps(args["num_sampling_steps"], device=device)
        elif args["sample_method"] == 'HeunDiscrete':  ########
            scheduler = HeunDiscreteScheduler()
        elif args["sample_method"] == 'EulerAncestralDiscrete':
            scheduler = EulerAncestralDiscreteScheduler()
        elif args["sample_method"] == 'DEISMultistep':
            scheduler = DEISMultistepScheduler()
        elif args["sample_method"] == 'KDPM2AncestralDiscrete':  #########
            scheduler = KDPM2AncestralDiscreteScheduler()
        print('videogen_pipeline', device)
        videogen_pipeline = VideoGenPipeline(vae=vae,
                                            text_encoder=text_encoder,
                                            tokenizer=tokenizer,
                                            scheduler=scheduler,
                                            transformer=transformer_model).to(device=device)
        # videogen_pipeline.enable_xformers_memory_efficient_attention()

        # video_grids = []
        if not isinstance(args["text_prompt"], list):
            args["text_prompt"] = [args["text_prompt"]]
        if len(args["text_prompt"]) == 1 and args["text_prompt"][0].endswith('txt'):
            text_prompt = open(args["text_prompt"][0], 'r').readlines()
            args["text_prompt"] = [i.strip() for i in text_prompt]
        for prompt in args["text_prompt"]:
            print('Processing the ({}) prompt'.format(prompt))
            prompt_save = prompt.replace(' ', '_')[:100]

            latents_dir = f"results/opensora_fifo/latents/{args["num_sampling_steps"]}steps/{prompt_save}"
            if args["version"] == "221x512x512":
                latents_dir = latents_dir.replace("opensora_fifo", "opensora_fifo_221")

            if args["output_dir"] is None:
                output_dir = f"results/opensora_fifo/video/{prompt_save}"

                if args["new_video_length"] != 100:
                    output_dir = output_dir.replace(f"{prompt_save}", f"{prompt_save}/{args["new_video_length"]}frames")
                if not args["lookahead_denoising"]:
                    output_dir = output_dir.replace(f"{prompt_save}", f"{prompt_save}/lookahead_denoising")
                if not args["num_partitions"] != 8:
                    output_dir = output_dir.replace(f"{prompt_save}", f"{prompt_save}/{args["num_partitions"]}partitions")    

                if args["version"] == "221x512x512":
                    output_dir = output_dir.replace("opensora_fifo", "opensora_fifo_221")
            else:
                output_dir = args["output_dir"]

            print("The results will be saved in", output_dir)
            print("The latents will be saved in", latents_dir)
            
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(latents_dir, exist_ok=True)

            is_run_base = not os.path.exists(os.path.join(latents_dir, "video.pt"))
            if is_run_base:
                videos = videogen_pipeline(prompt,
                                        num_frames=args["num_frames"],
                                        height=args["height"],
                                        width=args["width"],
                                        num_inference_steps=args["num_sampling_steps"],
                                        guidance_scale=args["guidance_scale"],
                                        enable_temporal_attentions=not args["force_images"],
                                        num_images_per_prompt=1,
                                        mask_feature=True,
                                        save_latents=True,
                                        latents_dir=latents_dir,
                                        return_dict=False,
                                        )

                output_path = os.path.join(output_dir, "origin.mp4")
                imageio.mimwrite(output_path, videos[0][0], fps=args["fps"], quality=9)  # highest quality is 10, lowest is 0
            
            videogen_pipeline.scheduler.set_timesteps(args["num_sampling_steps"], device=videogen_pipeline.text_encoder.device)
        
            latents = prepare_latents(args, latents_dir, scheduler=videogen_pipeline.scheduler)

            if args["save_frames"]:
                fifo_dir = os.path.join(output_dir, "fifo")
                os.makedirs(fifo_dir, exist_ok=True)
            
            fifo_video_frames = []
            fifo_first_latents = []

            timesteps = videogen_pipeline.scheduler.timesteps
            timesteps = torch.flip(timesteps, [0])
            if args["lookahead_denoising"]:
                timesteps = torch.cat([torch.full((args["video_length"]//2,), timesteps[0]).to(timesteps.device), timesteps])


            num_iterations = args["new_video_length"] + args["queue_length"] - args["video_length"] if args["version"] == "65x512x512" else args["new_video_length"] + args["queue_length"]
            for i in trange(num_iterations):
                num_inference_steps_per_gpu = args["video_length"]

                for rank in reversed(range(2 * args["num_partitions"] if args["lookahead_denoising"] else args["num_partitions"])):
                    if args["lookahead_denoising"]:
                        start_idx = (rank // 2) * num_inference_steps_per_gpu + (rank % 2) * (num_inference_steps_per_gpu // 2)
                    else:
                        start_idx = rank * num_inference_steps_per_gpu
                    midpoint_idx = start_idx + num_inference_steps_per_gpu // 2 + (rank % 2)
                    end_idx = start_idx + num_inference_steps_per_gpu

                    t = timesteps[start_idx:end_idx]
                    input_latents = latents[:,:,start_idx:end_idx].clone()

                    output_latents, first_latent, first_frame = videogen_pipeline.fifo_onestep(prompt,
                                            video_length=args["video_length"],
                                            height=args["height"],
                                            width=args["width"],
                                            num_inference_steps=args["num_sampling_steps"],
                                            guidance_scale=args["guidance_scale"],
                                            enable_temporal_attentions=not args["force_images"],
                                            num_images_per_prompt=1,
                                            mask_feature=True,
                                            latents=input_latents,
                                            timesteps=t,
                                            save_frames=args["save_frames"],
                                            )

                    if args["lookahead_denoising"]:
                        latents[:,:,midpoint_idx:end_idx] = output_latents[:,:,-(end_idx-midpoint_idx):]
                    else:
                        latents[:,:,start_idx:end_idx] = output_latents
                    del output_latents

                latents = shift_latents(latents, videogen_pipeline.scheduler)
                
                if args["save_frames"]:
                    output_path = os.path.join(fifo_dir, f"frame_{i:04d}.png")
                    imageio.mimwrite(output_path, first_frame, quality=9)  # highest quality is 10, lowest is 0

                fifo_first_latents.append(first_latent)

            num_vae = args["new_video_length"] // (args["video_length"]-1)
            
            if args["version"] == "65x512x512":
                first_idx = args["queue_length"] - args["video_length"]
            else:
                first_idx = args["queue_length"]

            fifo_vae_video_frames = []
            for i in range(num_vae):
                target_latents = torch.cat(fifo_first_latents[first_idx+i*(args["video_length"]-1):first_idx+(i+1)*(args["video_length"]-1)+1], dim=2)
                video = videogen_pipeline.decode_latents(target_latents)[0]

                if i == 0:
                    fifo_vae_video_frames.append(video)
                else:
                    fifo_vae_video_frames.append(video[1:])
            
            if num_vae > 0:
                fifo_vae_video_frames = torch.cat(fifo_vae_video_frames, dim=0)
                if args["output_dir"] is None:
                    output_vae_path = os.path.join(output_dir, "fifo_vae.mp4")
                else:
                    output_vae_path = os.path.join(args["output_dir"], f"{prompt_save}.mp4")
                imageio.mimwrite(output_vae_path, fifo_vae_video_frames, fps=args["fps"], quality=9)
        

    @torch.no_grad()
    def call_lvdm(
            self,
            prompt: Union[str, List[str]] = None,
            negative_prompt: str = "",
            num_inference_steps: int = 20,
            timesteps: List[int] = None,
            guidance_scale: float = 4.5,
            num_images_per_prompt: Optional[int] = 1,
            num_frames: Optional[int] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            clean_caption: bool = True,
            mask_feature: bool = True,
            enable_temporal_attentions: bool = True,
            save_latents: bool = False,
            latents_dir: str = None
    ) -> Union[VideoPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            mask_feature (`bool` defaults to `True`): If set to `True`, the text embeddings will be masked.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # 1. Check inputs. Raise error if not correct
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, negative_prompt, callback_steps, prompt_embeds, negative_prompt_embeds
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.text_encoder.device or self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
            mask_feature=mask_feature,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs[" TODO"]: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        # if self.transformer.config.sample_size == 128:
        #     resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
        #     aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1)
        #     resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
        #     aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)
        #     added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                if prompt_embeds.ndim == 3:
                    prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
                # if prompt_attention_mask.ndim == 2:
                #     prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input, # torch.size([2, 4, 17, 64, 64])
                    encoder_hidden_states=prompt_embeds, # torch.size([2, 25, 2096])
                    timestep=current_timestep, # torch.size([2]), e.g. [996, 996]
                    added_cond_kwargs=added_cond_kwargs,# {'resolution': None, 'aspect_ratio': None}
                    enable_temporal_attentions=enable_temporal_attentions, # True
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        if save_latents:
            torch.save(latents, latents_dir+"/video.pt")

        if not output_type == 'latents':
            video = self.decode_latents(latents)
            # video = video[:, :num_frames, :height, :width]
        else:
            video = latents
            return VideoPipelineOutput(video=video)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return VideoPipelineOutput(video=video)
    
    @torch.no_grad()
    def fifo_onestep(
            self,
            prompt: Union[str, List[str]] = None,
            negative_prompt: str = "",
            num_inference_steps: int = 20,
            timesteps: Optional[torch.IntTensor] = None,
            guidance_scale: float = 4.5,
            num_images_per_prompt: Optional[int] = 1,
            video_length: Optional[int] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            clean_caption: bool = True,
            mask_feature: bool = True,
            enable_temporal_attentions: bool = True,
            rank: int = 0,
            save_frames: bool = False,
    ) -> Union[VideoPipelineOutput, Tuple]:
        # 1. Check inputs. Raise error if not correct
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, negative_prompt, callback_steps, prompt_embeds, negative_prompt_embeds
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.text_encoder.device or self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
            mask_feature=mask_feature,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        latent_channels = self.transformer.config.in_channels


        # 6. Prepare extra step kwargs[" TODO"]: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        # if self.transformer.config.sample_size == 128:
        #     resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
        #     aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1)
        #     resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
        #     aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)
        #     added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}


        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        current_timestep = timesteps # torch.Size([f])

        current_timestep = current_timestep[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        current_timestep = current_timestep.expand(latent_model_input.shape[0], -1) # torch.Size([2, f])
        
        if prompt_embeds.ndim == 3:
                    prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
        # predict noise model_output
        noise_pred = self.transformer(
            latent_model_input, # torch.size([2, 4, 17, 64, 64])
            encoder_hidden_states=prompt_embeds, # torch.size([2, 25, 2096])
            timestep=current_timestep, # torch.size([2, f])
            added_cond_kwargs=added_cond_kwargs,# {'resolution': None, 'aspect_ratio': None}
            enable_temporal_attentions=enable_temporal_attentions, # True
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # learned sigma
        if self.transformer.config.out_channels // 2 == latent_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        else:
            noise_pred = noise_pred

        # compute previous image: x_t -> x_t-1
        for i in range(video_length):
            latents[:,:,[i]] = self.scheduler.step(noise_pred[:,:,[i]], timesteps[i], latents[:,:,[i]], **extra_step_kwargs, return_dict=False)[0]
        
        first_latent = None
        first_frame = None
        if rank == 0:
            first_latent = latents[:,:,[0]]
            if save_frames:
                first_frame = self.decode_latents(first_latent)[0]
            else:
                first_frame = None

        return latents, first_latent, first_frame
