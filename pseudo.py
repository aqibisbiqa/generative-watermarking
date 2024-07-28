import torch
import numpy as np
import random
import copy
import functools
import tqdm
from torchvision.transforms.functional import pil_to_tensor
import os

# own files
import ecc
import utils

class Psyduck():
    def __init__(
            self, 
            pipe, 
            keys=(10, 11, 12), 
            timesteps=50, 
            debug=False, 
            save_images=True, 
            prompt="A photo of a cat"
    ):
        self.pipe = pipe
        self.keys = keys
        
        # for generation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.timesteps = timesteps
        self.video_timesteps = 25
        self.prompt = prompt
        
        # for input
        images_for_svd = [
            "input_sample.png",
            "bearded_man.jpg",
            "dog_run.jpg",
            "low_res_cat.jpg",
        ]
        self.input_image_location = f"svd_base_images/{images_for_svd[1]}"

        # for output
        self.save_images = save_images
        self.process_type = "pt" # ["pt", "pil", "na"]

        # for experiment
        self.iters = 0

        # for debugging
        self.debug = debug

    ################################################################################################################################
    # ENCODING METHODS
    ################################################################################################################################
    
    @torch.no_grad()
    def encode(self, m: str, verbose=False):
        self.iters += 1
        cls_name = self.pipe.__class__.__name__
        if cls_name == "StegoFIFOVideoDiffusionPipeline":
            # return self._encode_long_video(m, verbose)
            raise NotImplementedError("longvideo not yet supported")
        elif cls_name == "StegoStableVideoDiffusionPipeline":
            return self._encode_video(m, verbose)
        elif cls_name == "StegoStableDiffusionPipeline":
            return self._encode_latent(m, verbose)
        elif cls_name == "StegoDDIMPixelPipeline":
            return self._encode_pixel(m, verbose)
        else:
            raise AttributeError(f"the {cls_name} is not supported")
    
    def _encode_long_video(self, m: str, verbose=False):

        # Synchronize settings
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        # timesteps = self.timesteps
        timesteps = 25
        timesteps = self.video_timesteps
        
        s_churn = 1.0
        height, width = 512, 512
        # num_frames = self.pipe.unet.config.num_frames
        num_frames = 15
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

        # Post-processing
        pt_frames = self.pipe.video_processor.postprocess_video(video=frames, output_type="pt")
        pil_frames = self.pipe.video_processor.postprocess_video(video=frames, output_type="pil")[0]

        # Save optionally
        if self.save_images:
            gif_path = f"logging/videos/{self.iters}_encode_video.gif"
            pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], optimize=False, duration=100, loop=0)

        # Output processing
        if self.process_type == "pt":
            frames = pt_frames
        elif self.process_type == "pil":
            frames = pil_frames

        return frames

    def _encode_video(self, m: str, verbose=False):

        # Synchronize settings
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        # timesteps = self.timesteps
        timesteps = 25
        timesteps = self.video_timesteps
        
        s_churn = 1.0
        # height = 576
        # width = 1024
        height, width = 512, 512
        # num_frames = self.pipe.unet.config.num_frames
        num_frames = 15
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

        # Post-processing
        pt_frames = self.pipe.video_processor.postprocess_video(video=frames, output_type="pt")
        print(f"frames {frames.shape}")
        pil_frames = self.pipe.video_processor.postprocess_video(video=frames, output_type="pil")[0]

        # Save optionally
        if self.save_images:
            os.makedirs("logging/videos", exist_ok=True)
            gif_path = f"logging/videos/{self.iters}_encode_video.gif"
            pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], optimize=False, duration=100, loop=0)

        # Output processing
        if self.process_type == "pt":
            frames = pt_frames
        elif self.process_type == "pil":
            frames = pil_frames

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
            os.makedirs("logging/images/latent", exist_ok=True)
            save_path = f"logging/images/latent/{self.iters}_encode_latent.png"
            pil_img.save(save_path)

        # Output handling
        if self.process_type == "pt":
            img = pt_img
        elif self.process_type == "pil":
            img = pil_img
            
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
            os.makedirs("logging/images/pixel", exist_ok=True)
            save_path = f"logging/images/pixel/{self.iters}_encode_pixel.png"
            utils.process_pixel(img)[0].save(save_path)
        
        return img

    ################################################################################################################################
    # DECODING METHODS
    ################################################################################################################################
    @torch.no_grad()
    def decode(self, img, verbose=False):
        cls_name = self.pipe.__class__.__name__
        if cls_name == "StegoStableVideoDiffusionPipeline":
            return self._decode_video(img, verbose)
        elif cls_name == "StegoStableDiffusionPipeline":
            return self._decode_latent(img, verbose)
        elif cls_name == "StegoDDIMPixelPipeline":
            return self._decode_pixel(img, verbose)
        else:
            raise AttributeError(f"the {cls_name} is not supported")

    def _decode_video(self, frames, verbose=False):
        
        # Synchronize settings
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        # timesteps = self.timesteps
        timesteps = 25
        timesteps = self.video_timesteps

        s_churn = 1.0
        # height = 576
        # width = 1024
        height, width = 512, 512
        # num_frames = self.pipe.unet.config.num_frames
        num_frames = 15
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
        err_rate = 1 - utils.empirical_success_rates["video"]

        # VAE decode
        if needs_upcasting:
            self.pipe.vae.to(dtype=torch.float16)
        frames_0 = self.pipe.decode_latents(latents_0, num_frames, decode_chunk_size)
        frames_1 = self.pipe.decode_latents(latents_1, num_frames, decode_chunk_size)

        # Post-processing
        pt_frames_0 = self.pipe.video_processor.postprocess_video(video=frames_0, output_type="pt")
        pt_frames_1 = self.pipe.video_processor.postprocess_video(video=frames_1, output_type="pt")
        pil_frames_0 = self.pipe.video_processor.postprocess_video(video=frames_0, output_type="pil")[0]
        pil_frames_1 = self.pipe.video_processor.postprocess_video(video=frames_1, output_type="pil")[0]

        # Save optionally
        if self.save_images:
            gif_path = f"logging/videos/{self.iters}_decode_video_0.gif"
            pil_frames_0[0].save(gif_path, save_all=True, append_images=pil_frames_0[1:], optimize=False, duration=100, loop=0)
            gif_path = f"logging/videos/{self.iters}_decode_video_1.gif"
            pil_frames_1[0].save(gif_path, save_all=True, append_images=pil_frames_1[1:], optimize=False, duration=100, loop=0)

        # Output processing
        if self.process_type == "pt":
            frames_0 = pt_frames_0
            frames_1 = pt_frames_1
        elif self.process_type == "pil":
            frames_0 = pil_frames_0
            frames_1 = pil_frames_1
        
        ######################
        # Online phase       #
        ######################

        def _undo_processing(frames):
            if self.process_type == "pil":
                frames = torch.stack([pil_to_tensor(frame) for frame in frames]).to(torch.float16).to(self.device)
                frames = torch.unsqueeze(frames, 0) / 255
            if self.process_type in ["pil", "pt"]:
                frames = (frames - 0.5) * 2
                frames = frames.permute(0, 2, 1, 3, 4)
            return frames

        if self.process_type:
            frames = _undo_processing(frames)
            frames_0 = _undo_processing(frames_0)
            frames_1 = _undo_processing(frames_1)
        
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

        m = utils.decode_message_from_image_diffs(latents, latents_0, latents_1, "video", verbose)
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
        err_rate = 1 - utils.empirical_success_rates["latent"]

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

        m = utils.decode_message_from_image_diffs(latents, latents_0, latents_1, "latent", verbose)
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
        err_rate = 1 - utils.empirical_success_rates["pixel"]

        # Optionally save images
        if self.save_images: 
            utils.process_pixel(img_0)[0].save(f"logging/images/pixel/{self.iters}_decode_pixel_0.png")
            utils.process_pixel(img_1)[0].save(f"logging/images/pixel/{self.iters}_decode_pixel_1.png")

        ######################
        # Online phase       #
        ######################

        # Decoding
        m = utils.decode_message_from_image_diffs(img, img_0, img_1, "pixel", verbose)

        return m
    
    def _decode_message_from_image_diffs(self, img, img_0, img_1, err_rate, verbose=False):
        diffs_0 = torch.norm(img - img_0, dim=(0, 1))
        diffs_1 = torch.norm(img - img_1, dim=(0, 1))

        if self.debug:
            show = 5
            print(diffs_0[:show, :show])
            print(diffs_1[:show, :show])

        m_dec = torch.where(diffs_0 < diffs_1, 0, 1).cpu().detach().numpy().astype(int)
        if verbose: print("Message AFTER Transmission:", m_dec, sep="\n")
        m_dec = m_dec.flatten()
        return ecc.ecc_recover(m_dec, err_rate)