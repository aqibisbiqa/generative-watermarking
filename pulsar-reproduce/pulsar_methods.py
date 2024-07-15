import torch
import numpy as np
import random
import copy
import functools
import tqdm
from torchvision.transforms.functional import pil_to_tensor

# own files
import ecc
from rate_estimation import estimate_rate
import utils

class Pulsar():
    def __init__(self, pipe, keys=(10, 11, 12), timesteps=50, debug=False, save_images=True, prompt="A photo of a cat"):
        self.pipe = pipe
        self.timesteps = timesteps
        self.prompt = prompt
        
        self.keys = keys

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_images = save_images
        self.debug = debug
        self.iters = 0
        self.process_type = "pt" # ["pt", "pil", "unproc"]

        sample_images = [
            "input_sample.png",
            "bearded_man.jpg",
            "dog_run.jpg",
            "low_res_cat.jpg",
        ]

        self.input_image_location = f"logging/images/for_svd/{sample_images[3]}"

    ################################################################################################################################
    # ENCODING METHODS
    ################################################################################################################################
    
    @torch.no_grad()
    def encode(self, m: str, verbose=False):
        self.iters += 1
        cls_name = self.pipe.__class__.__name__
        match cls_name:
            case "StegoStableVideoDiffusionPipeline":
                return self._encode_video(m, verbose)
            case "StegoStableDiffusionPipeline":
                return self._encode_latent(m, verbose)
            case "StegoDDIMPipeline":
                return self._encode_pixel(m, verbose)
            
            case "StableVideoDiffusionPipeline":
                return self._encode_video_old(m, verbose)
            case "StableDiffusionPipeline":
                return self._encode_latent_old(m, verbose)
            case "DDIMPipeline":
                return self._encode_pixel_old(m, verbose)
            case _:
                raise AttributeError(f"the {cls_name} is not supported")
    
    def _encode_video(self, m: str, verbose=False):

        # Synchronize settings
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        # timesteps = self.timesteps
        timesteps = 25
        
        s_churn = 1.0
        # height = 576
        # width = 1024
        height, width = 512, 512
        num_frames = self.pipe.unet.config.num_frames
        decode_chunk_size = num_frames // 4
        fps = 7
        motion_bucket_id = 127
        noise_aug_strength = 0.02
        num_videos_per_prompt = 1
        batch_size = 1

        image = utils.prepare_image(self.input_image_location, height, width)
        needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast

        # Conduct pipeline
        pipeline_output = self.pipe(
            stego_type="encode",
            payload=m,
            keys = self.keys,
            output_type="latent",
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=timesteps,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=g_k_s,
            return_dict=True,
        )

        latents = pipeline_output["frames"]

        # VAE decode
        if needs_upcasting:
            self.pipe.vae.to(dtype=torch.float16)
        frames = self.pipe.decode_latents(latents, num_frames, decode_chunk_size)

        # Save optionally
        if self.save_images:
            gif_path = f"logging/videos/{self.iters}_encode_video.gif"
            processed_frames = self.pipe.video_processor.postprocess_video(video=frames, output_type="pil")[0]
            processed_frames[0].save(gif_path, save_all=True, append_images=processed_frames[1:], optimize=False, duration=100, loop=0)

        return frames

    def _encode_video_old(self, m: str, verbose=False):

        # Synchronize settings
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps
        
        s_churn = 1.0
        # height = 576
        # width = 1024
        height, width = 512, 512
        num_frames = self.pipe.unet.config.num_frames
        decode_chunk_size = num_frames
        fps = 7
        motion_bucket_id = 127
        noise_aug_strength = 0.02
        num_videos_per_prompt = 1
        batch_size = 1

        # Initialize nonlocals for later
        latents = None
        image = utils.prepare_image(self.input_image_location, height, width)

        # For latent models, use callback to interact with denoising loop
        def _enc_callback(pipe, step_index, timestep, callback_kwargs):
            
            ##################
            # Offline Phase
            ##################
            
            # Interrupt denoising loop with two steps left
            if step_index != pipe.num_timesteps - 3:
                return callback_kwargs

            # The T-2'th denoising step is done, we do the rest manually
            pipe._interrupt = True

            print(f"beginning encode callback")

            # Make variables nonlocal so they can be referenced outside this callback
            nonlocal latents
            nonlocal image
            nonlocal fps
            
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            
            print(f"calculating image embeddings")

            image_embeddings = pipe._encode_image(image, self.device, num_videos_per_prompt=1, do_classifier_free_guidance=pipe.do_classifier_free_guidance)

            print(f"image_embeddings is {image_embeddings.shape}")

            fps = fps - 1

            # 4. Encode input image using VAE
            image = pipe.video_processor.preprocess(image, height=height, width=width).to(self.device)

            print(f"after preprocessing, image is {image.shape}")

            noise = torch.randn(image.shape, generator=g_k_s, dtype=image.dtype).to(self.device)
            image = image + noise_aug_strength * noise

            needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
            if needs_upcasting:
                pipe.vae.to(dtype=torch.float32)
            
            ### need to ensure this is synchronized ###
            image_latents = pipe._encode_vae_image(
                image,
                device=self.device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            )
            image_latents = image_latents.to(image_embeddings.dtype)

            # cast back to fp16 if needed
            if needs_upcasting:
                pipe.vae.to(dtype=torch.float16)

            print(f"after vae, image_latents is {image_latents.shape}")

            # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

            print(f"after unsqueezing, image_latents is {image_latents.shape}")
            
            added_time_ids = pipe._get_add_time_ids(
                fps,
                motion_bucket_id,
                noise_aug_strength,
                image_embeddings.dtype,
                batch_size,
                num_videos_per_prompt,
                pipe.do_classifier_free_guidance,
            )
            added_time_ids = added_time_ids.to(self.device)

            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################

            latents = callback_kwargs["latents"]

            # Estimate rate
            rate = estimate_rate(self, latents)
            
            #################
            # Online Phase
            #################

            print(f"online phase start, image_embeddings {image_embeddings.shape}, image latents {image_latents.shape}")

            # Perform T-1'th denoising step (g_k_0 and g_k_1)
            step_index += 1
            timestep = pipe.scheduler.timesteps[-2]  ### PENULTIMATE STEP ###

                    # predict noise
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

                    # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # sample two latents (g_k_0 and g_k_1)
            print(f"sched, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            
            latents_0 = pipe.scheduler.step(noise_pred, timestep, latents, s_churn=s_churn, generator=g_k_0).prev_sample
            print(f"sched_0, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            pipe.scheduler._step_index = None   # setting to None will allow us to use timestep properly

            latents_1 = pipe.scheduler.step(noise_pred, timestep, latents, s_churn=s_churn, generator=g_k_1).prev_sample
            print(f"sched_1, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            pipe.scheduler._step_index = None
            
            print(f"mixing shapes {latents_0.shape} and {latents_1.shape}")
            show = 4
            print(m[:show])
            print(latents_0[0, 0, 0, :show, :show])
            print(latents_1[0, 0, 0, :show, :show])

            # Encode payload and use it to mix the two latents 
            latents[:, :, :] = utils.mix_samples_using_payload(m, rate, latents_0, latents_1, self.device, verbose)
            
            print(latents[0, 0, 0, :show, :show])

            # Perform T'th denoising step (deterministic)
            step_index += 1
            timestep = pipe.scheduler.timesteps[-1]  ### LAST STEP ###

                    # predict noise
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

                    # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # sample final latent (determinstic)
            print(f"sched, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
            print(f"sched, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            pipe.scheduler._step_index = None
            
            # Exit callback by returning new latent w/ encoded latents
            callback_kwargs["latents"] = latents
            return callback_kwargs
        
        # Conduct pipeline
        _ = self.pipe(
            image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=timesteps,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=g_k_s,
            callback_on_step_end=_enc_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        # VAE decode
        frames = self.pipe.decode_latents(latents, num_frames, decode_chunk_size)
        
        print(f"after decode_latents, frames is {frames.shape}")

        # Save optionally
        if self.save_images:
            # self.pipe.video_processor.postprocess_video(video=frames, output_type="pil")[0].save("logging/images/encode_video.mp4")
            pass
        
        print(f"sending frames {len(frames), frames[0].shape}")

        return frames

    def _encode_latent(self, m: str, verbose=False):

        # Synchronize settings
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        print(f"sending {m[:10]}")
        
        # Conduct pipeline
        pipeline_output = self.pipe(
            stego_type="encode",
            payload=m,
            keys = self.keys,
            output_type="latent",
            prompt=self.prompt,
            num_inference_steps=timesteps,
            generator=g_k_s,
            return_dict=True,
        )

        latents = pipeline_output["images"]

        # VAE decode
        img = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]

        # Image processing
        pt_img = self.pipe.image_processor.postprocess(img, output_type="pt")
        pil_img = self.pipe.image_processor.postprocess(img, output_type="pil")[0]
        
        # Save optionally
        if self.save_images: 
            pil_img.save(f"logging/images/latent/{self.iters}_encode_latent.png")

        # Output handling
        if self.process_type == "pt":
            img = pt_img
        elif self.process_type == "pil":
            img = pil_img
            
        return img

    def _encode_latent_old(self, m: str, verbose=False):

        # Synchronize settings
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        # Initialize nonlocals for later
        latents = None

        # For latent models, use callback to interact with denoising loop
        def _enc_callback(pipe, step_index, timestep, callback_kwargs):
            
            ##################
            # Offline Phase
            ##################
            
            # Interrupt denoising loop with two steps left
            if step_index != pipe.num_timesteps - 3:
                return callback_kwargs

            # The T-2'th denoising step is done, we do the rest manually
            pipe._interrupt = True

            # Make variables nonlocal so they can be referenced outside this callback
            nonlocal latents

            # Extra kwargs :(
                # SD requires funky code to change generator, TODO: write PR to fix
            # pipe.generator = g_k_0 # does this work??
            extra_step_kwargs_s = pipe.prepare_extra_step_kwargs(g_k_s, eta)
            extra_step_kwargs_0 = pipe.prepare_extra_step_kwargs(g_k_0, eta)
            extra_step_kwargs_1 = pipe.prepare_extra_step_kwargs(g_k_1, eta)
            
            latents = callback_kwargs["latents"]
            prompt_embeds = callback_kwargs["prompt_embeds"]
            timestep_cond = None
            added_cond_kwargs = None

            # Estimate rate
            rate = estimate_rate(self, latents)
            
            #################
            # Online Phase
            #################

            # Perform T-1'th denoising step (g_k_0 and g_k_1)
            step_index += 1
            timestep = pipe.scheduler.timesteps[-2]  ### PENULTIMATE STEP ###

                    # predict noise
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

                    # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
                    # sample two latents (g_k_0 and g_k_1)
            latents_0 = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_0, return_dict=False)[0]
            latents_1 = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_1, return_dict=False)[0]
            
            # Encode payload and use it to mix the two latents 
            latents[:, :] = utils.mix_samples_using_payload(m, rate, latents_0, latents_1, self.device, verbose)
            
            # Perform T'th denoising step (deterministic)
            step_index += 1
            timestep = pipe.scheduler.timesteps[-1]  ### LAST STEP ###

                    # predict noise
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

                    # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # sample final latent (determinstic)
            latents = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_s, return_dict=False)[0]
            
            # Exit callback by returning new latent w/ encoded latents
            callback_kwargs["latents"] = latents
            return callback_kwargs
        
        # Conduct pipeline
        _ = self.pipe(
            self.prompt,
            num_inference_steps=timesteps,
            generator=g_k_s,
            callback_on_step_end=_enc_callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
        ).images[0]

        # VAE decode
        img = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]

        # Save optionally
        if self.save_images: 
            self.pipe.image_processor.postprocess(img, output_type="pil")[0].save(f"logging/images/latent/{self.iters}_encode_latent.png")
        
        return img

    def _encode_pixel(self, m: str, verbose=False):

        # Synchronize settings
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        pipeline_output = self.pipe(
            eta=eta,
            num_inference_steps=timesteps,
            output_type="pt",
            return_dict=True,
            stego_type="encode",
            keys=self.keys,
            payload=m,
        )

        img = pipeline_output["images"]

        # Optionally save image
        if self.save_images: 
            utils.process_pixel(img)[0].save(f"logging/images/pixel/{self.iters}_encode_pixel.png")
        
        return img
    
    def _encode_pixel_old(self, m: str, verbose=False):
        
        #################
        # Offline phase #
        #################

        # Synchronize settings
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        model = self.pipe.unet
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(timesteps)
        device = self.device

        image_shape = (
            1, 
            model.config.in_channels, 
            model.config.sample_size, 
            model.config.sample_size
        )
        samp = torch.randn(image_shape, generator=g_k_s, dtype=model.dtype).to(device)
        
        # Perform first T-2 denoising steps (g_k_s)
        for i, t in enumerate(tqdm.tqdm(scheduler.timesteps[:-2])):
            residual = model(samp, t).sample
            samp = scheduler.step(residual, t, samp, generator=g_k_s, eta=eta).prev_sample
            if verbose and ((timesteps-3-i) % 5 == 0):
                utils.display_sample(samp, i + 1)

        # Estimate rate
        rate = estimate_rate(self, samp)

        ##################
        #  Online phase  #
        ##################

        # Perform T-1'th denoising step (g_k_0 and g_k_1)
        t = scheduler.timesteps[-2]  ### PENULTIMATE STEP ###
        residual = model(samp, t).sample
        samp_0 = scheduler.step(residual, t, samp, generator=g_k_0, eta=eta).prev_sample
        samp_1 = scheduler.step(residual, t, samp, generator=g_k_1, eta=eta).prev_sample

        # Encode payload and use it to mix the two samples pixelwise
        samp[:, :] = utils.mix_samples_using_payload(m, rate, samp_0, samp_1, self.device, verbose)

        # Perform T'th denoising step (deterministic)
        t = scheduler.timesteps[-1]  ### LAST STEP ###
        residual = model(samp, t).sample
        img = scheduler.step(residual, t, samp).prev_sample

        # Optionally save image
        if self.save_images: utils.process_pixel(img)[0].save(f"logging/images/pixel/{self.iters}_encode_pixel.png")
        
        return img
    
    def _mix_samples_using_payload(self, payload, rate, samp_0, samp_1, verbose=False):
        m_ecc = ecc.ecc_encode(payload, rate)
        m_ecc.resize(samp_0[0, 0].shape, refcheck=False)
        if verbose: print("### Message BEFORE Transmission ###", m_ecc, "#"*35, sep="\n")
        m_ecc = torch.from_numpy(m_ecc).to(self.device)
        return torch.where(m_ecc == 0, samp_0[:, :], samp_1[:, :])

    ################################################################################################################################
    # DECODING METHODS
    ################################################################################################################################
    @torch.no_grad()
    def decode(self, img, verbose=False):
        cls_name = self.pipe.__class__.__name__
        match cls_name:
            case "StegoStableVideoDiffusionPipeline":
                return self._decode_video(img, verbose)
            case "StegoStableDiffusionPipeline":
                return self._decode_latent(img, verbose)
            case "StegoDDIMPipeline":
                return self._decode_pixel(img, verbose)
            
            case "StableVideoDiffusionPipeline":
                return self._decode_video_old(img, verbose)
            case "StableDiffusionPipeline":
                return self._decode_latent_old(img, verbose)
            case "DDIMPipeline":
                return self._decode_pixel_old(img, verbose)
            case _:
                raise AttributeError(f"the {cls_name} is not supported")

    def _decode_video(self, frames, verbose=False):
        
        # Synchronize settings
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        # timesteps = self.timesteps
        timesteps = 25

        s_churn = 1.0
        # height = 576
        # width = 1024
        height, width = 512, 512
        num_frames = self.pipe.unet.config.num_frames
        decode_chunk_size = num_frames // 4
        fps = 7
        motion_bucket_id = 127
        noise_aug_strength = 0.02
        num_videos_per_prompt = 1
        batch_size = 1
        
        image = utils.prepare_image(self.input_image_location, height, width)
        needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast
        
        # Conduct pipeline
        pipeline_output = self.pipe(
            stego_type="decode",
            keys = self.keys,
            output_type="latent",
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=timesteps,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=g_k_s,
            return_dict=True,
        )

        latents_0, latents_1 = pipeline_output["frames"].chunk(2)
        rate = pipeline_output["rate"]

        # VAE decode
        if needs_upcasting:
            self.pipe.vae.to(dtype=torch.float16)
        frames_0 = self.pipe.decode_latents(latents_0, num_frames, decode_chunk_size)
        frames_1 = self.pipe.decode_latents(latents_1, num_frames, decode_chunk_size)

        # Save optionally
        if self.save_images:
            gif_path = f"logging/videos/{self.iters}_decode_video_0.gif"
            processed_frames_0 = self.pipe.video_processor.postprocess_video(video=frames_0, output_type="pil")[0]
            processed_frames_0[0].save(gif_path, save_all=True, append_images=processed_frames_0[1:], optimize=False, duration=100, loop=0)
            gif_path = f"logging/videos/{self.iters}_decode_video_1.gif"
            processed_frames_1 = self.pipe.video_processor.postprocess_video(video=frames_1, output_type="pil")[0]
            processed_frames_1[0].save(gif_path, save_all=True, append_images=processed_frames_1[1:], optimize=False, duration=100, loop=0)

        ######################
        # Online phase       #
        ######################
        
        # Undo VAE (via VAE encode)
        def _invert_vae(frames, sample_mode="mode"):
            # frames.shape [batch_size, channels, num_frames, height, width]
            # -> [batch_size*num_frames, channels, height, width]
            frames = frames.permute(0, 2, 1, 3, 4)
            frames = torch.flatten(frames, 0, 1).to(torch.float16)

            print(f"before encoding, frames is {frames.shape}")

            # -> [batch_size*num_frames, num_channels_latents // 2, height // self.vae_scale_factor, width // self.vae_scale_factor]
            if sample_mode == "sample":
                frame_to_latent = lambda frame : self.pipe.vae.encode(frame).latent_dist.sample(g_k_s) * self.pipe.vae.config.scaling_factor
            elif sample_mode == "mode":
                frame_to_latent = lambda frame : self.pipe.vae.encode(frame).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            else:
                frame_to_latent = lambda frame : self.pipe.vae.encode(frame).latents * self.pipe.vae.config.scaling_factor
            frames = frame_to_latent(frames)
            frames = frames.permute((1, 0, 2, 3))
            frames = torch.unsqueeze(frames, 0)
            return frames

        print(f"before vae inversion, frames_0 is {frames_0.shape}")
        show = 4
        if self.debug or True:
            print(frames[0, 0, 0, :show, :show].numpy(force=True))
            print(frames_0[0, 0, 0, :show, :show].numpy(force=True))
            print(frames_1[0, 0, 0, :show, :show].numpy(force=True))
        
        latents = _invert_vae(frames)
        latents_0 = _invert_vae(frames_0)
        latents_1 = _invert_vae(frames_1)

        print(f"after invert_vae, latents_0 is {latents_0.shape}")
        
        assert latents.shape == latents_0.shape == latents_1.shape

            # debugging
        if self.debug or True:
            print(latents[0, 0, 0, :show, :show].numpy(force=True))
            print(latents_0[0, 0, 0, :show, :show].numpy(force=True))
            print(latents_1[0, 0, 0, :show, :show].numpy(force=True))

        m = self._decode_message_from_image_diffs(latents, latents_0, latents_1, rate, verbose)
        return m

    def _decode_video_old(self, frames, verbose=False):
        
        # Synchronize settings
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        s_churn = 1.0
        # height = 576
        # width = 1024
        height, width = 512, 512
        num_frames = self.pipe.unet.config.num_frames
        decode_chunk_size = num_frames
        fps = 7
        motion_bucket_id = 127
        noise_aug_strength = 0.02
        num_videos_per_prompt = 1
        batch_size = 1
        
        # Initialize nonlocals for later
        latents_0 = None
        latents_1 = None
        rate = None
        image = utils.prepare_image(self.input_image_location, height, width)

        # For latent models, use callback to get latents prior to vae decode
        def _dec_callback(pipe, step_index, timestep, callback_kwargs):
            
            ##################
            # Offline Phase
            ##################

            # Interrupt denoising loop with two steps left
            if step_index != pipe.num_timesteps - 3:
                return callback_kwargs

            # The T-2'th denoising step is done, we do the rest manually
            pipe._interrupt = True

            # Make variables nonlocal so they can be referenced outside this callback
            nonlocal latents_0
            nonlocal latents_1
            nonlocal rate
            nonlocal image
            nonlocal fps

            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################

            print(f"calculating image embeddings")
            
            image_embeddings = self.pipe._encode_image(image, self.device, num_videos_per_prompt=1, do_classifier_free_guidance=self.pipe.do_classifier_free_guidance)

            print(f"image_embeddings is {image_embeddings.shape}")
            
            fps = fps - 1

            # 4. Encode input image using VAE
            image = self.pipe.video_processor.preprocess(image, height=height, width=width).to(self.device)

            print(f"after preprocessing, image is {image.shape}")
            
            noise = torch.randn(image.shape, generator=g_k_s, dtype=image.dtype).to(self.device)
            image = image + noise_aug_strength * noise

            needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast
            if needs_upcasting:
                self.pipe.vae.to(dtype=torch.float32)
            
            ### need to ensure this is synchronized ###
            image_latents = self.pipe._encode_vae_image(
                image,
                device=self.device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.pipe.do_classifier_free_guidance,
            )
            image_latents = image_latents.to(image_embeddings.dtype)

            # cast back to fp16 if needed
            if needs_upcasting:
                self.pipe.vae.to(dtype=torch.float16)

            print(f"after vae, image_latents is {image_latents.shape}")
            
            # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

            added_time_ids = self.pipe._get_add_time_ids(
                fps,
                motion_bucket_id,
                noise_aug_strength,
                image_embeddings.dtype,
                batch_size,
                num_videos_per_prompt,
                self.pipe.do_classifier_free_guidance,
            )
            added_time_ids = added_time_ids.to(self.device)

            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            #######################################################
            
            latents = callback_kwargs["latents"]

            # Estimate rate
            rate = estimate_rate(self, latents)

            # Perform T-1'th denoising step (g_k_0 and g_k_1)
            step_index += 1
            timestep = pipe.scheduler.timesteps[-2]  ### PENULTIMATE STEP ###
            
                    # predict noise
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

                    # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # sample two latents (g_k_0 and g_k_1)
            print(f"sched, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            
            latents_0 = pipe.scheduler.step(noise_pred, timestep, latents, s_churn=s_churn, generator=g_k_0).prev_sample
            print(f"sched_0, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            pipe.scheduler._step_index = None
            
            latents_1 = pipe.scheduler.step(noise_pred, timestep, latents, s_churn=s_churn, generator=g_k_1).prev_sample
            print(f"sched_1, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            pipe.scheduler._step_index = None

            show = 4
            print(latents_0[0, 0, 0, :show, :show])
            print(latents_1[0, 0, 0, :show, :show])
            
            # Perform T'th denoising step (deterministic)
            step_index += 1
            timestep = pipe.scheduler.timesteps[-1]  ### LAST STEP ###
            
                    # predict noise using latents_0 and latents_1
            latent_model_input_0 = torch.cat([latents_0] * 2) if pipe.do_classifier_free_guidance else latents_0
            latent_model_input_0 = pipe.scheduler.scale_model_input(latent_model_input_0, timestep)
            latent_model_input_0 = torch.cat([latent_model_input_0, image_latents], dim=2)
            noise_pred_0 = pipe.unet(
                latent_model_input_0,
                timestep,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

            latent_model_input_1 = torch.cat([latents_1] * 2) if pipe.do_classifier_free_guidance else latents_1
            latent_model_input_1 = pipe.scheduler.scale_model_input(latent_model_input_1, timestep)
            latent_model_input_1 = torch.cat([latent_model_input_1, image_latents], dim=2)
            noise_pred_1 = pipe.unet(
                latent_model_input_1,
                timestep,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

                    # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond_0, noise_pred_cond_0 = noise_pred_0.chunk(2)
                noise_pred_0 = noise_pred_uncond_0 + pipe.guidance_scale * (noise_pred_cond_0 - noise_pred_uncond_0)
                noise_pred_uncond_1, noise_pred_cond_1 = noise_pred_1.chunk(2)
                noise_pred_1 = noise_pred_uncond_1 + pipe.guidance_scale * (noise_pred_cond_1 - noise_pred_uncond_1)

                    # sample final latents (deterministic)
            print(f"sched, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            
            latents_0 = pipe.scheduler.step(noise_pred_0, timestep, latents_0).prev_sample
            print(f"sched_0, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            pipe.scheduler._step_index = None
            
            latents_1 = pipe.scheduler.step(noise_pred_1, timestep, latents_1).prev_sample
            print(f"sched_1, timestep {timestep}, sched step {pipe.scheduler._step_index}")
            pipe.scheduler._step_index = None
            
            # We will do the VAE step manually, so exit callback with anything
            callback_kwargs["latents"] = torch.zeros_like(latents)
            return callback_kwargs
        
        # Conduct pipeline
        _ = self.pipe(
            image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=timesteps,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=g_k_s,
            callback_on_step_end=_dec_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        print(f"after decode callback, latents_0 is {latents_0.shape}")

        # VAE decode
        frames_0 = self.pipe.decode_latents(latents_0, num_frames, decode_chunk_size)
        frames_1 = self.pipe.decode_latents(latents_1, num_frames, decode_chunk_size)
        # frames_0 = self.pipe.vae.decode(latents_0 / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]
        # frames_1 = self.pipe.vae.decode(latents_1 / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]

        print(f"after decode_latents, frames_0 is {frames_0.shape}")

        # Save optionally
        if self.save_images:
            # self.pipe.video_processor.postprocess_video(video=frames_0, output_type="pil")[0].save("logging/images/decode_video_0.mp4")
            pass
            # self.pipe.video_processor.postprocess_video(video=frames_1, output_type="pil")[0].save("logging/images/decode_video_1.mp4")
            pass

        ######################
        # Online phase       #
        ######################
        
        # Undo VAE (via VAE encode)
        def _invert_vae(frames, sample_mode="mode"):
            # frames.shape [batch_size, channels, num_frames, height, width]
            # -> [batch_size*num_frames, channels, height, width]
            frames = frames.permute(0, 2, 1, 3, 4)
            frames = torch.flatten(frames, 0, 1).to(torch.float16)

            # -> [batch_size*num_frames, num_channels_latents // 2, height // self.vae_scale_factor, width // self.vae_scale_factor]
            if sample_mode == "sample":
                frames_to_latent = lambda frames : self.pipe.vae.encode(frames).latent_dist.sample(g_k_s) * self.pipe.vae.config.scaling_factor
            elif sample_mode == "mode":
                frames_to_latent = lambda frames : self.pipe.vae.encode(frames).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            else:
                frames_to_latent = lambda frames : self.pipe.vae.encode(frames).latents * self.pipe.vae.config.scaling_factor
            return frames_to_latent(frames)

        print(f"before vae inversion, frames_0 is {frames_0.shape}")
        show = 4
        if self.debug or True:
            print(frames[0, 0, 0, :show, :show].numpy(force=True))
            print(frames_0[0, 0, 0, :show, :show].numpy(force=True))
            print(frames_1[0, 0, 0, :show, :show].numpy(force=True))
        
        latents = _invert_vae(frames)
        latents_0 = _invert_vae(frames_0)
        latents_1 = _invert_vae(frames_1)

        print(f"after invert_vae, latents_0 is {latents_0.shape}")
        
        assert latents.shape == latents_0.shape == latents_1.shape

            # debugging
        if self.debug or True:
            print(latents[0, 0, :show, :show].numpy(force=True))
            print(latents_0[0, 0, :show, :show].numpy(force=True))
            print(latents_1[0, 0, :show, :show].numpy(force=True))

        m = self._decode_message_from_image_diffs(latents[:1], latents_0[:1], latents_1[:1], rate, verbose)
        return m

    def _decode_latent(self, img, verbose=False):
        
        # Synchronize settings
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        # Conduct pipeline
        pipeline_output = self.pipe(
            stego_type="decode",
            keys = self.keys,
            output_type="latent",
            prompt=self.prompt,
            num_inference_steps=timesteps,
            generator=g_k_s,
            return_dict=True,
        )

        latents_0, latents_1 = pipeline_output["images"].chunk(2)
        rate = pipeline_output["rate"]

        # VAE decode
        img_0 = self.pipe.vae.decode(latents_0 / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]
        img_1 = self.pipe.vae.decode(latents_1 / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]

        # Image processing
        pt_img_0 = self.pipe.image_processor.postprocess(img_0, output_type="pt")
        pt_img_1 = self.pipe.image_processor.postprocess(img_1, output_type="pt")
        pil_img_0 = self.pipe.image_processor.postprocess(img_0, output_type="pil")[0]
        pil_img_1 = self.pipe.image_processor.postprocess(img_1, output_type="pil")[0]

        # Save optionally
        if self.save_images:
            pil_img_0.save(f"logging/images/latent/{self.iters}_decode_latent_0.png")
            pil_img_1.save(f"logging/images/latent/{self.iters}_decode_latent_1.png")
        
        # Output processing
        if self.process_type == "pt":
            img_0 = pt_img_0
            img_1 = pt_img_1
        elif self.process_type == "pil":
            img_0 = pil_img_0
            img_1 = pil_img_1

        ######################
        # Online phase       #
        ######################

        def _undo_processing(image):
            if self.process_type == "pil":
                image = torch.unsqueeze(pil_to_tensor(image).to(torch.float16).to(self.device), 0) / 255
                image = (image - 0.5) * 2
            elif self.process_type == "pt":
                image = (image - 0.5) * 2
            return image

        if self.process_type:
            img = _undo_processing(img)
            img_0 = _undo_processing(img_0)
            img_1 = _undo_processing(img_1)
        
        # Undo VAE (via VAE encode)
        def _invert_vae(image, sample_mode="sample"):
            if sample_mode == "sample":
                img_to_latent = lambda image : self.pipe.vae.encode(image).latent_dist.sample(g_k_s) * self.pipe.vae.config.scaling_factor
            elif sample_mode == "mode":
                img_to_latent = lambda image : self.pipe.vae.encode(image).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            else:
                img_to_latent = lambda image : self.pipe.vae.encode(image).latents * self.pipe.vae.config.scaling_factor
            return img_to_latent(image)

        latents = _invert_vae(img)
        latents_0 = _invert_vae(img_0)
        latents_1 = _invert_vae(img_1)
        
        assert latents.shape == latents_0.shape == latents_1.shape

            # debugging
        if self.debug or True:
            print(latents[0, 0, 0, :10].numpy(force=True))
            print(latents_0[0, 0, 0, :10].numpy(force=True))
            print(latents_1[0, 0, 0, :10].numpy(force=True))

        m = self._decode_message_from_image_diffs(latents, latents_0, latents_1, rate, verbose)
        return m

    def _decode_latent_old(self, img, verbose=False):
        
        # Synchronize settings
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        # Initialize nonlocals for later
        latents_0 = None
        latents_1 = None
        rate = None

        # For latent models, use callback to get latents prior to vae decode
        def _dec_callback(pipe, step_index, timestep, callback_kwargs):
            
            ##################
            # Offline Phase
            ##################

            # Interrupt denoising loop with two steps left
            if step_index != pipe.num_timesteps - 3:
                return callback_kwargs

            # The T-2'th denoising step is done, we do the rest manually
            pipe._interrupt = True

            # Make variables nonlocal so they can be referenced outside this callback
            nonlocal latents_0
            nonlocal latents_1
            nonlocal rate

            # Extra kwargs :(
            # SD requires funky code to change generator, TODO: write PR to fix
            pipe.generator = g_k_0 # does this work??
            extra_step_kwargs_s = pipe.prepare_extra_step_kwargs(g_k_s, eta)
            extra_step_kwargs_0 = pipe.prepare_extra_step_kwargs(g_k_0, eta)
            extra_step_kwargs_1 = pipe.prepare_extra_step_kwargs(g_k_1, eta)
            
            latents = callback_kwargs["latents"]
            prompt_embeds = callback_kwargs["prompt_embeds"]
            timestep_cond = None
            added_cond_kwargs = None

            # Estimate rate
            rate = estimate_rate(self, latents)

            # Perform T-1'th denoising step (g_k_0 and g_k_1)
            step_index += 1
            timestep = pipe.scheduler.timesteps[-2]  ### PENULTIMATE STEP ###
            
                    # predict noise
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
            noise_pred = pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

                    # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # sample two latents (g_k_0 and g_k_1)
            latents_0 = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_0, return_dict=False)[0]
            latents_1 = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_1, return_dict=False)[0]
            
            # Perform T'th denoising step (deterministic)
            step_index += 1
            timestep = pipe.scheduler.timesteps[-1]  ### LAST STEP ###
            
                    # predict noise using latents_0 and latents_1
            latent_model_input_0 = torch.cat([latents_0] * 2) if pipe.do_classifier_free_guidance else latents_0
            latent_model_input_0 = pipe.scheduler.scale_model_input(latent_model_input_0, timestep)
            noise_pred_0 = pipe.unet(
                latent_model_input_0,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            latent_model_input_1 = torch.cat([latents_1] * 2) if pipe.do_classifier_free_guidance else latents_1
            latent_model_input_1 = pipe.scheduler.scale_model_input(latent_model_input_1, timestep)
            noise_pred_1 = pipe.unet(
                latent_model_input_1,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

                    # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond_0, noise_pred_text_0 = noise_pred_0.chunk(2)
                noise_pred_0 = noise_pred_uncond_0 + pipe.guidance_scale * (noise_pred_text_0 - noise_pred_uncond_0)
                noise_pred_uncond_1, noise_pred_text_1 = noise_pred_1.chunk(2)
                noise_pred_1 = noise_pred_uncond_1 + pipe.guidance_scale * (noise_pred_text_1 - noise_pred_uncond_1)

                    # sample final latents (deterministic)
            latents_0 = pipe.scheduler.step(noise_pred_0, timestep, latents_0, **extra_step_kwargs_s, return_dict=False)[0]
            latents_1 = pipe.scheduler.step(noise_pred_1, timestep, latents_1, **extra_step_kwargs_s, return_dict=False)[0]
            
            # We will do the VAE step manually, so exit callback with anything
            callback_kwargs["latents"] = torch.zeros_like(latents)
            return callback_kwargs
        
        # Conduct pipeline
        _ = self.pipe(
            self.prompt,
            num_inference_steps=timesteps,
            generator=g_k_s,
            callback_on_step_end=_dec_callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
        ).images[0]

        # VAE decode
        img_0 = self.pipe.vae.decode(latents_0 / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]
        img_1 = self.pipe.vae.decode(latents_1 / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]

        # Save optionally
        if self.save_images:
            self.pipe.image_processor.postprocess(img_0, output_type="pil")[0].save(f"logging/images/latent/{self.iters}_decode_latent_0.png")
            self.pipe.image_processor.postprocess(img_1, output_type="pil")[0].save(f"logging/images/latent/{self.iters}_decode_latent_1.png")

        ######################
        # Online phase       #
        ######################
        
        # Undo VAE (via VAE encode)
        def _invert_vae(image, sample_mode="sample"):
            # image = pil_to_tensor(image).to(torch.float16).to(self.device)
            # image = torch.unsqueeze(image, 0)
            if sample_mode == "sample":
                img_to_latent = lambda image : self.pipe.vae.encode(image).latent_dist.sample(g_k_s) * self.pipe.vae.config.scaling_factor
            elif sample_mode == "mode":
                img_to_latent = lambda image : self.pipe.vae.encode(image).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            else:
                img_to_latent = lambda image : self.pipe.vae.encode(image).latents * self.pipe.vae.config.scaling_factor
            return img_to_latent(image)

        latents = _invert_vae(img)
        latents_0 = _invert_vae(img_0)
        latents_1 = _invert_vae(img_1)
        
        assert latents.shape == latents_0.shape == latents_1.shape

            # debugging
        if self.debug or True:
            print(latents[0, 0, 0, :10].numpy(force=True))
            print(latents_0[0, 0, 0, :10].numpy(force=True))
            print(latents_1[0, 0, 0, :10].numpy(force=True))

        m = self._decode_message_from_image_diffs(latents, latents_0, latents_1, rate, verbose)
        return m

    def _decode_pixel(self, img, verbose=False):


        ######################
        # Offline phase      #
        ######################

        # Synchronize encode/decode settings
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        pipeline_output = self.pipe(
            eta=eta,
            num_inference_steps=timesteps,
            output_type="pt",
            return_dict=True,
            stego_type="decode",
            keys=self.keys,
        )
        
        img_0, img_1 = pipeline_output["images"].chunk(2)
        rate = pipeline_output["rate"]

        # Optionally save images
        if self.save_images: 
            utils.process_pixel(img_0)[0].save(f"logging/images/pixel/{self.iters}_decode_pixel_0.png")
            utils.process_pixel(img_1)[0].save(f"logging/images/pixel/{self.iters}_decode_pixel_1.png")

        ######################
        # Online phase       #
        ######################

        # Decoding
        m = self._decode_message_from_image_diffs(img, img_0, img_1, rate, verbose)

        return m
    
    def _decode_pixel_old(self, img, verbose=False):

        ######################
        # Offline phase      #
        ######################

        # Synchronize encode/decode settings
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        model = self.pipe.unet
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(timesteps)
        device = self.device

        image_shape = (
            1, 
            model.config.in_channels, 
            model.config.sample_size, 
            model.config.sample_size
        )
        samp = torch.randn(image_shape, generator=g_k_s, dtype=model.dtype).to(device)

        for i, t in enumerate(tqdm.tqdm(scheduler.timesteps[:-2])):
            residual = model(samp, t).sample
            samp = scheduler.step(residual, t, samp, generator=g_k_s, eta=eta).prev_sample
            if verbose and ((i + 1) % 5 == 0):
                utils.display_sample(samp, i + 1)

        rate = estimate_rate(self, samp)

        t = scheduler.timesteps[-2]   # penultimate step
        residual = model(samp, t).sample
        samp_0 = scheduler.step(residual, t, samp, generator=g_k_0, eta=eta).prev_sample
        samp_1 = scheduler.step(residual, t, samp, generator=g_k_1, eta=eta).prev_sample

        t = scheduler.timesteps[-1]   # last step
        residual_0 = model(samp_0, t).sample
        residual_1 = model(samp_1, t).sample
        img_0 = scheduler.step(residual_0, t, samp_0, eta=eta).prev_sample
        img_1 = scheduler.step(residual_1, t, samp_1, eta=eta).prev_sample

        # Optionally save images
        if self.save_images: utils.process_pixel(img_0)[0].save(f"logging/images/pixel/{self.iters}_decode_pixel_0.png")
        if self.save_images: utils.process_pixel(img_1)[0].save(f"logging/images/pixel/{self.iters}_decode_pixel_1.png")

        ######################
        # Online phase       #
        ######################

        # Decoding
        m = self._decode_message_from_image_diffs(img, img_0, img_1, rate, verbose)

        return m
    
    def _decode_message_from_image_diffs(self, img, img_0, img_1, rate, verbose=False):
        diffs_0 = torch.norm(img - img_0, dim=(0, 1))
        diffs_1 = torch.norm(img - img_1, dim=(0, 1))

        if self.debug:
            show = 5
            print(diffs_0[:show, :show])
            print(diffs_1[:show, :show])

        m_dec = torch.where(diffs_0 < diffs_1, 0, 1).cpu().detach().numpy().astype(int)
        if verbose: print("Message AFTER Transmission:", m_dec, sep="\n")
        m_dec = m_dec.flatten()
        return ecc.ecc_recover(m_dec, rate)