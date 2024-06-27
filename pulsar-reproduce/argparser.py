import argparse

default_model = "church"
default_iters = 10
default_steps = 50

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=default_model, help="\n".join([\
        "which model to steganographically embed into \n\n",
        "\"church:\" (DDIMPipeline, google/ddpm-church-256)",
        "\"bedroom:\" (DDIMPipeline, google/ddpm-bedroom-256)",
        "\"cat:\" (DDIMPipeline, google/ddpm-cat-256)",
        "\"celeb:\" (DDIMPipeline, google/ddpm-celebahq-256)",
        "\"butterflies:\" (DDIMPipeline, dboshardy/ddim-butterflies-128)",
        "\"lego:\" (DDIMPipeline, lukasHoel/ddim-model-128-lego-diffuse-1000)",
        "\"sd15:\" (StableDiffusionPipeline, runwayml/stable-diffusion-v1-5)",
        "\"sd21:\" (StableDiffusionPipeline, stabilityai/stable-diffusion-2-1-base)",
        "\"sd21real:\" (StableDiffusionPipeline, friedrichor/stable-diffusion-2-1-realistic)",
    ]))
    parser.add_argument('--iters', type=int, default=default_iters, help='number of iterations to test model')
    parser.add_argument('--timesteps', type=int, default=default_steps, help='number of timesteps for denoising loop')
    parser.add_argument('--filename', type=str, default="results.txt", help='output file')
    parser.add_argument('--verbose', type=bool, default=False, help='toggle to show more feedback during inference')
    parser.add_argument('--ignore_warnings', type=bool, default=True,
                        help='whether to ignore DeprecationWarnings and FutureWarnings')
    return parser