from sys import argv
import argparse

# own files

def main(args):

    ### Imports ###
    import torch
    import numpy as np
    import utils

    if args.ignore_warnings:
        print("ignoring warnings")
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
    
    from psyduck import Psyduck
    from supported_models import get_pipeline

    ### Experiment Setup ###
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise Exception("use gpu sir")

    pipe = get_pipeline(args.model, device)
    
    ### Experiment Loop ###
    accs, i = [], 0
    np.random.seed(0)
    p = Psyduck(pipe)
    while i < args.iters:
        try:
            print("#"*75)
            k = np.random.randint(1000, size=(3,))
            p.keys = k

            m_sz = 7680
            m = np.random.randint(256, size=m_sz, dtype=np.uint8)
            
            # prompt = "A man with a mustache."
            # prompt = "A photo of a cat."
            # p.prompt = prompt
            prompts = [
                "A man with a mustache.",
                "A photo of a cat.",
                "Eiffel Tower under the blue sky.",
                "Sydney Opera House.",
                "Leaning Tower of Pisa.",
                "Young girl with blond hair.",
                "A cute rabbit.",
                "Obama giving a speech.",
                "Tomatoes hanging on a tree.",
                "A multi layered sandwich.",
            ]
            p.prompt = prompts[np.random.randint(len(prompts))]
            
            p.iters += 1
            print(f"Iteration {i+1} using keys {k}")
            
            if args.gen_covers:
                print("GENERATING COVER SAMPLE")
                p.generate_cover()
            
            print("ENCODING")
            img = p.encode(m)
            
            print("DECODING")
            out = p.decode(img)
        except ValueError as e:
            if "operands could not be broadcast together with shapes" in str(e):
                print("stupid broadcast error, retrying")
            else:
                raise
        except ZeroDivisionError as e:
            print("stupid galois field error, retrying")
        else:
            print(f"length of m is {len(m)} bytes")
            print(f"length of out is {len(out)} bytes")
            acc = utils.calc_acc(m, out)
            accs.append(acc)
            print(f"Run accuracy {acc:.2%}")
            i += 1
            # plot pareto curves
    
    torch.cuda.empty_cache()
    
    ### Print Output ###
    print("#"*75)
    print(f"Final Average Accuracy {np.mean(accs):.2%} +/- {np.std(accs):.2%}")
    print(f"{np.round(accs, 2)}")
    return accs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="sd15", help="\n".join([\
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
    parser.add_argument('--iters', type=int, default=1, help='number of iterations to test model')
    parser.add_argument('--img_timesteps', type=int, default=50, help='number of denoising steps for image models')
    parser.add_argument('--vid_timesteps', type=int, default=25, help='number of denoising steps for video models')
    parser.add_argument('--filename', type=str, default="results.txt", help='output file')
    parser.add_argument('--ignore_warnings', type=bool, default=True,
                        help='whether to ignore DeprecationWarnings and FutureWarnings')
    parser.add_argument('--gen_covers', action='store_true', help='to generate cover images w/out steganographic embeddings')

    args = parser.parse_args()

    accs = main(args)
    # TODO: log accs