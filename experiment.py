from sys import argv
import argparse
import os
from prettytable import PrettyTable

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
    p = Psyduck(pipe)
    
    ### Logging Setup ###
    log_headers = ["model_type", "model", "div_steps", "iters", "bytes_enc", "acc", "std"]
    log_path = f"logging/{p.model_type}/{args.model}.txt"
    print(f"using {log_path}")
    if not os.path.exists(log_path):
        with open(log_path, 'w') as log:
            log.write(",".join(log_headers))
    log_data = []
    
    ### Experiment Loop ###
    for div_steps in [1]:
        accs, i = [], 0
        np.random.seed(args.seed)
        p.div_steps = div_steps
        p.iters = 180
        while i < args.iters:
            try:
                print("#"*75)
                k = np.random.randint(1000, size=(3,))
                p.keys = k

                m_sz = 3200
                m = np.random.randint(256, size=m_sz, dtype=np.uint8)
                
                # provide random context
                if p.model_type in ["latent", "longvideo"]:
                    prompts = [
                        "A man with a mustache.",
                        "A photo of a cat.",
                        "Eiffel Tower under the blue sky.",
                        "Sydney Opera House.",
                        "Leaning Tower of Pisa.",
                        "Young girl with blond hair.",
                        "A cute rabbit.",
                        "Tomatoes hanging on a tree.",
                        "A multi layered sandwich.",
                    ]
                    p.prompt = prompts[np.random.randint(len(prompts))]
                elif p.model_type in ["video"]:
                    svd_base_imgs = [
                        "bearded_man.jpg",
                        "dog_run.jpg",
                        "low_res_cat.jpg",
                    ]
                    svd_base_img = svd_base_imgs[np.random.randint(len(svd_base_imgs))]
                    p.input_image_location = f"svd_base_images/{svd_base_img}"
                
                p.iters += 1
                print(f"Iteration {i+1} using keys {k}")
                
                if args.generate_covers:
                    print("GENERATING COVER SAMPLE")
                    p.generate_cover()
                    i += 1
                    continue

                print("ENCODING")
                img = p.encode(m)
                
                print("DECODING")
                out = p.decode(img)
            except ValueError as e:
                if "operands could not be broadcast together with shapes" in str(e):
                    print("stupid broadcast error, retrying")
                    p.iters -= 1
                else:
                    raise
            except ZeroDivisionError as e:
                print("stupid galois field error, retrying")
                p.iters -= 1
            else:
                print(f"length of m is {len(m)} bytes")
                print(f"length of out is {len(out)} bytes")
                acc = utils.calc_acc(m, out)
                accs.append(acc)
                print(f"Run accuracy {acc:.2%}")
                i += 1
        
        ## recall that log_headers = ["model_type", "model", "div_steps", "iters", "bytes_enc", "acc", "std"]
        log_row = [p.model_type, args.model, div_steps, args.iters, min(m_sz, len(out)), np.mean(accs).round(4), np.std(accs).round(4)]
        with open(log_path, 'a') as log:
            log.write("\n")
            log.write(",".join([str(datum) for datum in log_row]))
        log_data.append(log_row)
        
        print("#"*75)
        print(f"Final Average Accuracy for {args.model} w/ {div_steps} div_steps over {args.iters} runs:",
              f"{np.mean(accs):.2%} +/- {np.std(accs):.2%}")
        print(f"{np.round(accs, 2)}")
        
    ### Print Output ###
    table = PrettyTable()
    table.field_names = log_headers
    table.add_rows(log_data)
    print("\n"+"#"*75, "\n")
    print(table)

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
    parser.add_argument('--seed', type=int, default=0, help='seed for key and random message generation')
    parser.add_argument('--img_timesteps', type=int, default=50, help='number of denoising steps for image models')
    parser.add_argument('--vid_timesteps', type=int, default=25, help='number of denoising steps for video models')
    parser.add_argument('--filename', type=str, default="results.txt", help='output file')
    parser.add_argument('--ignore_warnings', type=bool, default=True,
                        help='whether to ignore DeprecationWarnings and FutureWarnings')
    parser.add_argument('--generate_covers', action='store_true', help='to generate cover images w/out steganographic embeddings')

    args = parser.parse_args()

    accs = main(args)