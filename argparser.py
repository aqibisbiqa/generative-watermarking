import argparse

default_model = "church"
default_iters = 1
default_steps = 50

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=default_model, help="\n".join([\
        "which model to steganographically embed into \n\n",
        "\"church\": google/ddpm-church-256)",
        "\"bedroom\": google/ddpm-bedroom-256)",
        "\"cat\": google/ddpm-cat-256)",
        "\"celeb\": google/ddpm-celebahq-256)",
        "\"butterflies\": dboshardy/ddim-butterflies-128)",
        "\"lego\": lukasHoel/ddim-model-128-lego-diffuse-1000)",
        "\"sd15\": runwayml/stable-diffusion-v1-5)",
        "\"sd21\": stabilityai/stable-diffusion-2-1-base)",
        "\"sd21real\": friedrichor/stable-diffusion-2-1-realistic)",
        "\"svd\": stabilityai/stable-video-diffusion-img2vid",
        "\"svdxt\": stabilityai/stable-video-diffusion-img2vid-xt",
    ]))
    parser.add_argument('--iters', type=int, default=default_iters, help='number of iterations to test model')
    parser.add_argument('--timesteps', type=int, default=default_steps, help='number of timesteps for denoising loop')
    parser.add_argument('--filename', type=str, default="results.txt", help='output file')
    parser.add_argument('--verbose', type=bool, default=False, help='toggle to show more feedback during inference')
    parser.add_argument('--ignore_warnings', type=bool, default=True,
                        help='whether to ignore DeprecationWarnings and FutureWarnings')
    parser.add_argument('--gen_covers', action='store_true', help='to generate cover images w/out steganographic embeddings')
    return parser