from typing import List, Optional, Tuple, Union

import torch

from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from utils import mix_samples_using_payload, process_pixel, empirical_success_rates

class StegoDDIMPixelPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        stego_type: str,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        keys: tuple = (10, 11, 12),
        payload = None,
    ) -> Union[ImagePipelineOutput, Tuple]:

        match stego_type:
            case "encode":
                assert payload is not None
            case "decode":
                assert payload is None
            case _:
                raise AttributeError("stego_type must be one of [\"encode\", \"decode\"]")

        g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in keys])
        device = self._execution_device

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        image = randn_tensor(image_shape, generator=g_k_s, device=self._execution_device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            model_output = self.unet(image, t).sample
            image = self.scheduler.step(
                model_output, t, image, eta=eta, generator=g_k_s, use_clipped_model_output=use_clipped_model_output, 
            ).prev_sample
            if i == num_inference_steps-3: break
        
        # Perform T-1'th denoising step (g_k_0 and g_k_1)
        t = self.scheduler.timesteps[-2]  ### PENULTIMATE STEP ###
        resid = self.unet(image, t).sample
        image_0 = self.scheduler.step(resid, t, image, generator=g_k_0, eta=eta).prev_sample
        image_1 = self.scheduler.step(resid, t, image, generator=g_k_1, eta=eta).prev_sample

        if stego_type == "encode":
            # Encode payload and use it to mix the two samples pixelwise
            image[:, :] = mix_samples_using_payload(payload, image_0, image_1, model_type="pixel")

            # Perform T'th denoising step (deterministic)
            t = self.scheduler.timesteps[-1]  ### LAST STEP ###
            resid = self.unet(image, t).sample
            image = self.scheduler.step(resid, t, image).prev_sample
        elif stego_type == "decode":
            t = self.scheduler.timesteps[-1]  ### LAST STEP ###
            resid_0 = self.unet(image_0, t).sample
            resid_1 = self.unet(image_1, t).sample
            image_0 = self.scheduler.step(resid_0, t, image_0, eta=eta).prev_sample
            image_1 = self.scheduler.step(resid_1, t, image_1, eta=eta).prev_sample
            image = torch.cat([image_0, image_1])

        if output_type == "pil":
            image = process_pixel(image)

        if not return_dict:
            return (image, )

        return {"images": image}
