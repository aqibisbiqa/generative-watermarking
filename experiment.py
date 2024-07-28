from sys import argv
from argparser import argument_parser

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
    
    from pseudo import Psyduck
    from supported_models import get_pipeline

    ### Experiment Setup ###
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise Exception("use gpu sir")

    pipe = get_pipeline(args.model, device, args.old)
    
    ### Experiment Loop ###
    accs, i = [], 0
    np.random.seed(0)
    p = Psyduck(pipe)
    while i < args.iters:
        try:
            print("#"*75)
            # img_sz = pipe.unet.config.sample_size
            # m_sz = (img_sz**2 // 512) * 25
            k = np.random.randint(1000, size=(3,))
            # m_sz = 10000
            # m_sz = 1536 # pixel
            m_sz = 1440 # svd
            # m_sz = 96 # sd15, sd21
            m = np.random.randint(256, size=m_sz, dtype=np.uint8)
            print(f"Iteration {i+1} using keys {k}")
            prompt = "A man with a mustache."
            # prompt = "A photo of a cat."
            p.keys = k
            p.prompt = prompt
            print("ENCODING")
            img = p.encode(m, verbose=args.verbose)
            print("DECODING")
            out = p.decode(img, verbose=args.verbose)
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

if __name__ == '__main__':
    _args = argument_parser().parse_args()
    accs = main(_args)
    # TODO: log accs