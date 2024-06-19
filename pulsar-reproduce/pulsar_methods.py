import torch
import numpy as np
import random
import copy
import functools
import tqdm

# own files
from ecc import *
from rate_estimation import *
from utils import *

class Pulsar():
    def __init__(self, pipe, keys=(10, 11, 12), timesteps=50, debug=False):
        self.pipe = pipe
        self.timesteps = timesteps
        
        self.keys = keys
        # k_s, k_0, k_1 = keys
        # g_k_s, g_k_0, g_k_1 = torch.Generator(), torch.Generator(), torch.Generator()
        # g_k_s.manual_seed(k_s)
        # g_k_0.manual_seed(k_0)
        # g_k_1.manual_seed(k_1)

        print(self.pipe.config)
        self.latent_model = "vae" in self.pipe.config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    ################################################################################################################################
    # ENCODING METHODS
    ################################################################################################################################
    
    @torch.no_grad()
    def encode(self, m: str, verbose=False):
        if self.latent_model:
            pass
        else:
            return self.encode_pixel(m, verbose)
    
    # For latent models, use callbacks to encode
    def encode_latent(self):
        eta = 1
        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in self.keys])
        timesteps = self.timesteps
        def enc_callback(pipe, step_index, timestep, callback_kwargs):
            # interrupt denoising loop with two steps left
            # stop_idx = pipe.num_timesteps - 2
            stop_idx = pipe.num_timesteps - 3
            if step_index != stop_idx: return callback_kwargs
            pipe._interrupt = True

            # SD requires funky code to change generator, TODO: write PR to fix
            pipe.generator = g_k_0 # does this work??
            extra_step_kwargs_s = pipe.prepare_extra_step_kwargs(g_k_s, eta)
            extra_step_kwargs_0 = pipe.prepare_extra_step_kwargs(g_k_0, eta)
            extra_step_kwargs_1 = pipe.prepare_extra_step_kwargs(g_k_1, eta)
            
            # SD Denoising Loop
            latents = callback_kwargs["latents"]
            latents = pipe.scheduler.scale_model_input(latents, timestep)
            noise_pred = pipe.unet(
                latents,
                timestep,
                # encoder_hidden_states=prompt_embeds,
                # timestep_cond=timestep_cond,
                # cross_attention_kwargs=self.cross_attention_kwargs,
                # added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute two latents using k_0 and k_1
            latents_0 = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_0, return_dict=False)[0]
            latents_1 = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_1, return_dict=False)[0]

            rate = 0
            sz = pipe.unet.config.sample_size
            m_ecc = ecc_encode(m, rate)
            m_ecc = np.reshape(m_ecc, (sz, sz))
            # if verbose: print("Message BEFORE Transmission:", m_ecc, sep="\n")
            timestep = pipe.scheduler.timesteps[-2]  # penultimate timestep
            for i in range(sz):
                for j in range(sz):
                    match m_ecc[i][j]:
                        case 0:
                            latents[:, :, i, j] = latents_0[:, :, i, j]
                        case 1:
                            latents[:, :, i, j] = latents_1[:, :, i, j]
            
            timestep = pipe.scheduler.timesteps[-1]  # last timestep
            noise_pred = pipe.unet(
                latents,
                t,
                # encoder_hidden_states=prompt_embeds,
                # timestep_cond=timestep_cond,
                # cross_attention_kwargs=self.cross_attention_kwargs,
                # added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs_s, return_dict=False)[0]
            
            callback_kwargs["latents"] = latents
            return callback_kwargs
        
        return self.pipe(
            "A photo of a cat",
            num_inference_steps=timesteps,
            generator=g_k_s,
            callback_on_step_end=enc_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

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

        # rate = estimate_rate(samp, k)
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
            pass
        else:
            return self.decode_pixel(img, verbose)

    def decode_pixel(self, img,verbose=False):

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
                # pos = sz * i + j
                n_0 = torch.norm(img[:, :, i, j] - img_0[:, :, i, j])
                n_1 = torch.norm(img[:, :, i, j] - img_1[:, :, i, j])
                if n_0 > n_1:
                    m_dec[i][j] = 1
        if verbose: print("Message AFTER Transmission:", m_dec, sep="\n")
        m_dec = m_dec.flatten()
        m = ecc_recover(m_dec, rate)
        return m