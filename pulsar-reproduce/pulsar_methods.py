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
        # k_s, k_0, k_1 = keys
        # g_k_s, g_k_0, g_k_1 = torch.Generator(), torch.Generator(), torch.Generator()
        # g_k_s.manual_seed(k_s)
        # g_k_0.manual_seed(k_0)
        # g_k_1.manual_seed(k_1)

        # print(self.pipe.config)
        self.latent_model = "vae" in self.pipe.config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_images = save_images
        self.debug = debug

    ################################################################################################################################
    # ENCODING METHODS
    ################################################################################################################################
    
    @torch.no_grad()
    def encode(self, m: str, verbose=False):
        if self.latent_model:
            return self._encode_latent(m, verbose)
        else:
            return self._encode_pixel(m, verbose)
    
    def _encode_latent(self, m: str, verbose=False):

        # Synchronize settings
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        # For latent models, use callback to interact with denoising loop
        def enc_callback(pipe, step_index, timestep, callback_kwargs):
            
            ##################
            # Offline Phase
            ##################
            
            # Interrupt denoising loop with two steps left
            if step_index != pipe.num_timesteps - 3:
                return callback_kwargs

            # The T-2'th denoising step is done, we do the rest manually
            pipe._interrupt = True

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
            latents[:, :] = self._mix_samples_using_payload(m, rate, latents_0, latents_1, verbose)
            
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
            
            # Exit callback by returning new latent w/ encoded message
            callback_kwargs["latents"] = latents
            return callback_kwargs
        
        # Generate stegoimage
        img = self.pipe(
            self.prompt,
            num_inference_steps=timesteps,
            generator=g_k_s,
            callback_on_step_end=enc_callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
        ).images[0]

        # Save optionally
        if self.save_images: img.save("logging/images/encode_latent.png")
        
        return img

    def _encode_pixel(self, m: str, verbose=False):
        
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
        samp[:, :] = self._mix_samples_using_payload(m, rate, samp_0, samp_1, verbose)

        # Perform T'th denoising step (deterministic)
        t = scheduler.timesteps[-1]  ### LAST STEP ###
        residual = model(samp, t).sample
        img = scheduler.step(residual, t, samp).prev_sample

        # Optionally save image
        if self.save_images: utils.process_pixel(img)[0].save("logging/images/encode_pixel.png")
        
        return img
    
    def _mix_samples_using_payload(self, payload, rate, samp_0, samp_1, verbose=False):
        m_ecc = ecc.ecc_encode(payload, rate)
        m_ecc.resize(samp_0[0, 0].shape)
        if verbose: print("### Message BEFORE Transmission ###", m_ecc, "#"*35, sep="\n")
        m_ecc = torch.from_numpy(m_ecc).to(self.device)
        return torch.where(m_ecc == 0, samp_0[:, :], samp_1[:, :])

    ################################################################################################################################
    # DECODING METHODS
    ################################################################################################################################
    @torch.no_grad()
    def decode(self, img, verbose=False):
        if self.latent_model:
            return self._decode_latent(img, verbose)
        else:
            return self._decode_pixel(img, verbose)

    def _decode_latent(self, img, verbose=False):
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        global latents_0
        global latents_1
        global rate

        # For latent models, use callback to get latents prior to vae decode
        def dec_callback(pipe, step_index, timestep, callback_kwargs):
            
            ##################
            # Offline Phase
            ##################

            # Interrupt denoising loop with two steps left
            if step_index != pipe.num_timesteps - 3:
                return callback_kwargs

            # The T-2'th denoising step is done, we do the rest manually
            pipe._interrupt = True

            # Use globals so they can be referenced outside this callback
            global latents_0
            global latents_1
            global rate

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
            callback_on_step_end=dec_callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
        ).images[0]

        # VAE decode + postprocessing
        img_0 = self.pipe.vae.decode(latents_0 / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]
        img_0 = self.pipe.image_processor.postprocess(img_0, output_type="pil")[0]
        img_1 = self.pipe.vae.decode(latents_1 / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]
        img_1 = self.pipe.image_processor.postprocess(img_1, output_type="pil")[0]

        # Save optionally
        if self.save_images: img_0.save("logging/images/decode_latent_0.png")
        if self.save_images: img_1.save("logging/images/decode_latent_1.png")

        ######################
        # Online phase       #
        ######################
        
        # Undo VAE (via VAE encode)
        def pil_to_latent(image, sample_mode="sample"):
            image = pil_to_tensor(image).to(torch.float16).to(self.device)
            image = torch.unsqueeze(image, 0)
            if sample_mode == "sample":
                img_to_latent = lambda image : self.pipe.vae.encode(image).latent_dist.sample(g_k_s) * self.pipe.vae.config.scaling_factor
            elif sample_mode == "mode":
                img_to_latent = lambda image : self.pipe.vae.encode(image).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            else:
                img_to_latent = lambda image : self.pipe.vae.encode(image).latents * self.pipe.vae.config.scaling_factor
            return img_to_latent(image)

        latents = pil_to_latent(img)
        latents_0 = pil_to_latent(img_0)
        latents_1 = pil_to_latent(img_1)
        
        assert latents.shape == latents_0.shape == latents_1.shape

            # debugging
        if self.debug or True:
            print(latents[0, 0, :3, :3])
            print(latents_0[0, 0, :3, :3])
            print(latents_1[0, 0, :3, :3])

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
        if self.save_images: utils.process_pixel(img_0)[0].save("logging/images/decode_pixel_0.png")
        if self.save_images: utils.process_pixel(img_1)[0].save("logging/images/decode_pixel_1.png")

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