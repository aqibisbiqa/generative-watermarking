import torch
import numpy as np
import random
import copy
import functools
import tqdm
import torchvision.transforms as transforms

# own files
from ecc import *
from rate_estimation import *
from utils import *

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

    ################################################################################################################################
    # ENCODING METHODS
    ################################################################################################################################
    
    @torch.no_grad()
    def encode(self, m: str, verbose=False):
        if self.latent_model:
            return self.encode_latent(m, verbose)
        else:
            return self.encode_pixel(m, verbose)
    
    def encode_latent(self, m: str, verbose=False):
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        # For latent models, use callback to encode
        def enc_callback(pipe, step_index, timestep, callback_kwargs):
            # interrupt denoising loop with two steps left
            # stop_idx = pipe.num_timesteps - 2
            latents = callback_kwargs["latents"]
            stop_idx = pipe.num_timesteps - 3
            if step_index != stop_idx: return callback_kwargs
            pipe._interrupt = True

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
            
            ###################
            # "Online" Phase
            ###################
            step_index += 1
            timestep = pipe.scheduler.timesteps[-2]  ### PENULTIMATE STEP ###
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

            # compute two latents using k_0 and k_1
            latents_0 = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_0, return_dict=False)[0]
            latents_1 = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_1, return_dict=False)[0]

            # print(f"{step_index} {timestep}: {latents_0.shape} latents_0")
            # print(f"{step_index} {timestep}: {latents_1.shape} latents_1")
            
            # rate = estimate_rate(latents, self.keys)
            rate = 0
            # print(pipe.unet.config)
            sz = pipe.unet.config.sample_size
            m_ecc = ecc_encode(m, rate)
            m_ecc = np.reshape(m_ecc, (sz, sz))
            if verbose: print("Message BEFORE Transmission:", m_ecc, sep="\n")
            for i in range(sz):
                for j in range(sz):
                    match m_ecc[i][j]:
                        case 0:
                            latents[:, :, i, j] = latents_0[:, :, i, j]
                        case 1:
                            latents[:, :, i, j] = latents_1[:, :, i, j]
            # print(f"{step_index} {timestep}: {latents.shape} latents")
            
            step_index += 1
            timestep = pipe.scheduler.timesteps[-1]  ### LAST STEP ###
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
            # print(f"{step_index} {timestep}: {latent_model_input.shape}")
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

            latents = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_s, return_dict=False)[0]
            
            callback_kwargs["latents"] = latents
            return callback_kwargs
        
        img = self.pipe(
            self.prompt,
            num_inference_steps=timesteps,
            generator=g_k_s,
            callback_on_step_end=enc_callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
        ).images[0]
        if self.save_images: img.save("encode_latent.png")
        return img

    def encode_pixel(self, m: str, verbose=False):
        
        ######################
        # Offline phase      #
        ######################
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
            if verbose and ((timesteps-3-i) % 5 == 0):
                display_sample(samp, i + 1)
        # print("OFFLINE SAMPLE:", samp[:, :, :5, :5], sep="\n")

        # rate = estimate_rate(samp, self.keys)
        rate = 0

        ######################
        # Online phase       #
        ######################
        sz = model.config.sample_size
        m_ecc = ecc_encode(m, rate)
        m_ecc = np.reshape(m_ecc, (sz, sz))
        if verbose: print("Message BEFORE Transmission:", m_ecc, sep="\n")

        t = scheduler.timesteps[-2]  # penultimate timestep

        # prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        # variance = scheduler._get_variance(t, prev_timestep)
        # print(f"t: {t}\nPREVIOUS TIMESTEP: {prev_timestep}\nVARIANCE: {variance}")

        residual = get_residual(model, samp, t)
        # torch.manual_seed(k_0) # is this necessary?
        samp_0 = scheduler.step(residual, t, samp, generator=g_k_0, eta=eta).prev_sample
        # print("\n\n SAMPLE 0:", samp_0[:, :, :3, :3], sep="\n")

        # torch.manual_seed(k_1) # is this necessary?
        samp_1 = scheduler.step(residual, t, samp, generator=g_k_1, eta=eta).prev_sample
        # print("\n\n SAMPLE 1:", samp_1[:, :, :3, :3], sep="\n")

        for i in range(sz):
            for j in range(sz):
                match m_ecc[i][j]:
                    case 0:
                        samp[:, :, i, j] = samp_0[:, :, i, j]
                    case 1:
                        samp[:, :, i, j] = samp_1[:, :, i, j]
        # print("\n\n PEN SAMPLE:", samp[:, :, :3, :3], sep="\n")

        t = scheduler.timesteps[-1]  # last timestep
        residual = get_residual(model, samp, t)
        img = scheduler.step(residual, t, samp).prev_sample
        # print("\n\n FINAL IMAGE:", img[:, :, :3, :3], sep="\n")
        return img
    


    ################################################################################################################################
    # DECODING METHODS
    ################################################################################################################################
    @torch.no_grad()
    def decode(self, img, verbose=False):
        if self.latent_model:
            return self.decode_latent(img, verbose)
        else:
            return self.decode_pixel(img, verbose)

    def decode_latent(self, img, verbose=False):
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps

        global latents_0
        global latents_1
        rate = None

        # For latent models, use callback to get latents prior to vae decode
        def dec_callback(pipe, step_index, timestep, callback_kwargs):
            # interrupt denoising loop with two steps left
            # stop_idx = pipe.num_timesteps - 2
            latents = callback_kwargs["latents"]
            global latents_0
            global latents_1
            stop_idx = pipe.num_timesteps - 3
            if step_index != stop_idx: return callback_kwargs
            pipe._interrupt = True

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
            
            ###################
            # "Online" Phase
            ###################
            step_index += 1
            timestep = pipe.scheduler.timesteps[-2]  ### PENULTIMATE STEP ###
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

            # compute two latents using k_0 and k_1
            latents_0 = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_0, return_dict=False)[0]
            latents_1 = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_1, return_dict=False)[0]
            
            # rate = estimate_rate(latents, self.keys)
            rate = 0
            
            step_index += 1
            timestep = pipe.scheduler.timesteps[-1]  ### LAST STEP ###
            latent_model_input_0 = torch.cat([latents_0] * 2) if pipe.do_classifier_free_guidance else latents_0
            latent_model_input_0 = pipe.scheduler.scale_model_input(latent_model_input_0, timestep)
            latent_model_input_1 = torch.cat([latents_1] * 2) if pipe.do_classifier_free_guidance else latents_1
            latent_model_input_1 = pipe.scheduler.scale_model_input(latent_model_input_1, timestep)
            
            noise_pred_0 = pipe.unet(
                latent_model_input_0,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

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

            latents_0 = pipe.scheduler.step(noise_pred_0, timestep, latents_0, **extra_step_kwargs_s, return_dict=False)[0]
            latents_1 = pipe.scheduler.step(noise_pred_1, timestep, latents_1, **extra_step_kwargs_s, return_dict=False)[0]
            
            callback_kwargs["latents"] = torch.zeros_like(latents)
            return callback_kwargs
        
        _ = self.pipe(
            self.prompt,
            num_inference_steps=timesteps,
            generator=g_k_s,
            callback_on_step_end=dec_callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
        ).images[0]

        # print(f"**type of latents_0 is {type(latents_0)}")

        img_0 = self.pipe.vae.decode(latents_0 / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]
        img_1 = self.pipe.vae.decode(latents_1 / self.pipe.vae.config.scaling_factor, return_dict=False, generator=g_k_s)[0]

        img_0 = self.pipe.image_processor.postprocess(img_0, output_type="pil")[0]
        img_1 = self.pipe.image_processor.postprocess(img_1, output_type="pil")[0]

        if self.save_images: img_0.save("decode_latent_0.png")
        if self.save_images: img_1.save("decode_latent_1.png")

        def pil_to_latent(image, sample_mode="sample"):
            image = transforms.functional.pil_to_tensor(image).to(torch.float16).to(self.device)
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

        print(latents[0, 0, :3, :3])
        print(latents_0[0, 0, :3, :3])
        print(latents_1[0, 0, :3, :3])

        sz = self.pipe.unet.config.sample_size
        m_dec = np.zeros((sz, sz), dtype=int)
        for i in range(sz):
            for j in range(sz):
                n_0 = torch.norm(latents[:, :, i, j] - latents_0[:, :, i, j])
                n_1 = torch.norm(latents[:, :, i, j] - latents_1[:, :, i, j])
                if n_0 > n_1:
                    m_dec[i][j] = 1
        if verbose: print("Message AFTER Transmission:", m_dec, sep="\n")
        m_dec = m_dec.flatten()
        m = ecc_recover(m_dec, rate)
        return m


    def decode_pixel(self, img, verbose=False):

        ######################
        # Offline phase      #
        ######################
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
            residual = get_residual(model, samp, t)
            samp = scheduler.step(residual, t, samp, generator=g_k_s, eta=eta).prev_sample
            if verbose and ((i + 1) % 5 == 0):
                display_sample(samp, i + 1)
        # print("OFFLINE SAMPLE:", samp[:, :, :5, :5], sep="\n")

        # rate = estimate_rate(samp, k)
        rate = 0

        t = scheduler.timesteps[-2]   # penultimate step
        residual = get_residual(model, samp, t)
        samp_0 = scheduler.step(residual, t, samp, generator=g_k_0, eta=eta).prev_sample
        samp_1 = scheduler.step(residual, t, samp, generator=g_k_1, eta=eta).prev_sample

        t = scheduler.timesteps[-1]   # last step
        residual_0 = get_residual(model, samp_0, t)
        residual_1 = get_residual(model, samp_1, t)
        img_0 = scheduler.step(residual_0, t, samp_0, eta=eta).prev_sample
        img_1 = scheduler.step(residual_1, t, samp_1, eta=eta).prev_sample

        ######################
        # Online phase       #
        ######################
        sz = model.config.sample_size
        m_dec = np.zeros((sz, sz), dtype=int)
        for i in range(sz):
            for j in range(sz):
                # print(img[:, :, i, j] - img_0[:, :, i, j])
                n_0 = torch.norm(img[:, :, i, j] - img_0[:, :, i, j])
                n_1 = torch.norm(img[:, :, i, j] - img_1[:, :, i, j])
                if n_0 > n_1:
                    m_dec[i][j] = 1
        if verbose: print("Message AFTER Transmission:", m_dec, sep="\n")
        m_dec = m_dec.flatten()
        m = ecc_recover(m_dec, rate)
        return m