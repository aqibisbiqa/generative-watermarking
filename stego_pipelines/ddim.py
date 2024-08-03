from typing import List, Optional, Tuple, Union

import torch

from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from utils import mix_samples_using_payload, process_pixel

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
        num_div_steps = 1,
    ) -> Union[ImagePipelineOutput, Tuple]:

        match stego_type:
            case "cover":
                pass
            case "encode":
                assert payload is not None
            case "decode":
                pass
            case _:
                raise AttributeError("stego_type must be one of [\"cover\", \"encode\", \"decode\"]")

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
        
        penul = num_inference_steps-2
        div_timesteps = torch.arange(penul, penul-num_div_steps, -1).flip(0)
        
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            resid = self.unet(image, t).sample
            if i not in div_timesteps or stego_type == "cover":
                image = self.scheduler.step(
                    resid, t, image, eta=eta, generator=g_k_s, use_clipped_model_output=use_clipped_model_output, 
                ).prev_sample
            else:
                if i == div_timesteps[0]:
                    # first divergent step
                    i0, i1 = image.clone(), image.clone()
                    r0, r1 = resid.clone(), resid.clone()
                else:
                    # already diverged
                    i0, i1 = image.chunk(2)
                    r0, r1 = resid.chunk(2)
                i0 = self.scheduler.step(
                    r0, t, i0, eta=eta, generator=g_k_s, use_clipped_model_output=use_clipped_model_output, 
                ).prev_sample
                i1 = self.scheduler.step(
                    r1, t, i1, eta=eta, generator=g_k_s, use_clipped_model_output=use_clipped_model_output, 
                ).prev_sample
                image = torch.cat([i0, i1])

                if i == penul:
                    if stego_type == "encode":
                        i0, i1 = image.chunk(2)
                        image = mix_samples_using_payload(payload, i0, i1, model_type="pixel")
                    elif stego_type == "decode":
                        pass
            
        if output_type == "pil":
            image = process_pixel(image)

        if not return_dict:
            return (image, )

        return {"images": image}
