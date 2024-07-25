# PSyDUCK: Perfectly Secure Steganographic Diffusion Models

Diffusion models offer a unique avenue for steganography (communicating secrets through mundane cover media).
We leverage the added noise in the late-stage denoising steps and the performance of modern VQ-VAEs to establish a reliable, high capacity steganographic channel.

The primary breakthrough of our method is that our steganographic embeddings are **perfectly secure**, in the sense that the unperturbed distribution of all output samples of the diffusion model and the distribution of samples with any encoded message are the same.
Thus, regardless of the number of leaked samples, an adversary will never be able to distinguish between a regular sample and that imbued with sensitive information.

NB: PSyDUCK is short for <u>P</u>erfectly-Secure <u>S</u>teganograph<u>y</u> via <u>D</u>iffusion model <u>U</u>sage to <u>C</u>ommunicate <u>K</u>overtly

## Usage

### Prerequisites


### Encode
```
python3 encode.py 
    --model     [supported_model] 
    --payload   [message_you_want_to_send]
    --keys      [(k_s, k_0, k_1)]
    --prompt    [prompt_if_txt_to_img]
    --baseimg   [path_to_base_image_if_using_img_to_###]
    --saveto    [path_to_save_sample]
```

### Decode
```
python3 encode.py 
    --model     [supported_model] 
    --message   [message_you_were_sent]
    --keys      [(k_s, k_0, k_1)]
    --prompt    [prompt_if_txt_to_img]
    --baseimg   [path_to_base_image_if_using_img_to_###]

```

### Run Experiment
```
python3 experiment.py
    --model     [supported_model] 
    --iters     [iterations_to_run]
    --prompt    [prompt_if_txt_to_img]
    --baseimg   [path_to_base_image_if_using_img_to_###]
```

## WIPs
- Associated paper/report
- Infinite video support (w/ FIFO-Diffusion)
- Web interface (w/ Gradio)
- Appealing acronym for our methods (open to suggestions :)