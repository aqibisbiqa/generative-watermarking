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
    
    from pulsar_methods import Pulsar
    from supported_models import get_pipeline

    ### Experiment Setup ###
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise Exception("use gpu sir")

    pipe = get_pipeline(args.model, device)
    # pipe = pipe.to(device)
    
    ### Experiment Loop ###
    accs, i = [], 0
    while i < args.iters:
        # try:
            print("#"*75)
            # img_sz = pipe.unet.config.sample_size
            # m_sz = (img_sz**2 // 512) * 25
            m_sz = 1500
            m = np.random.randint(256, size=m_sz, dtype=np.uint8)
            k = tuple(int(r) for r in np.random.randint(1000, size=(3,)))
            # k = (10, 11, 12)
            print(f"Iteration {i+1} using keys {k}")
            prompt = "Portrait photo of a man with mustache."
            p = Pulsar(pipe, k, args.timesteps, prompt=prompt)
            print("ENCODING")
            img = p.encode(m, verbose=args.verbose)
            print("DECODING")
            out = p.decode(img, verbose=args.verbose)
        # except ValueError:
        #     print("stupid broadcast error, retrying")
        # except ZeroDivisionError:
        #     print("stupid galois field error, retrying")
        # else:
            print(f"length of m is {len(m)} bytes")
            print(f"length of out is {len(out)} bytes")
            acc = utils.calc_acc(m, out)
            accs.append(acc)
            print(f"Run accuracy {acc:.2%}")
            i += 1
    
    torch.cuda.empty_cache()
    ### Print Output ###
    print("#"*75)
    print(f"Final Average Accuracy {np.mean(accs):.2%}")
    return accs

if __name__ == '__main__':
    _args = argument_parser().parse_args()
    accs = main(_args)
    # run_filename = f"./logging/results.txt"
    # append results per layer
        # with open(filename, 'a') as f:
        #     f.write(_net_class+' '+str(lr)+' '+str(h)+' '+str(l)+' '+str(comb_layers)+':'+line+'\n')