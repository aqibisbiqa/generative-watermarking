import torch

def get_pipeline(model, device):

    model_is_supported = False

    for model_type in models_by_type:
        if model in models_by_type[model_type]:
            model_is_supported = True
            repo = models_by_type[model_type][model]
        if model_is_supported:
            break
    if not model_is_supported:
        raise NotImplementedError(f"the {model} model is not yet supported")

    print(f"Will load {model} model from {repo}")
    match model_type:
        case "pixel":
            from stego_pipelines.ddim import StegoDDIMPixelPipeline
            pipe = StegoDDIMPixelPipeline.from_pretrained(repo)
            pipe.to(device)
        case "latent":
            from stego_pipelines.latent import StegoStableDiffusionPipeline
            pipe = StegoStableDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
            pipe.to(device)
        case "video":
            from stego_pipelines.video import StegoStableVideoDiffusionPipeline
            pipe = StegoStableVideoDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16, variant="fp16")
            pipe.to(device)
            # pipe.enable_sequential_cpu_offload()
            # pipe.enable_model_cpu_offload()
        case "longvideo":
            raise NotImplementedError("longvideo not yet supported")
    
    return pipe

models_by_type = {
    "latent": {
        "sd15": "runwayml/stable-diffusion-v1-5",
        "sd21": "stabilityai/stable-diffusion-2-1-base",
        "sd21real": "friedrichor/stable-diffusion-2-1-realistic",
    },

    "pixel": {
        "church": "google/ddpm-church-256",
        "bedroom": "google/ddpm-bedroom-256",
        "cat": "google/ddpm-cat-256",
        "celeb": "google/ddpm-celebahq-256",

        "butterflies": "dboshardy/ddim-butterflies-128",
        "lego": "lukasHoel/ddim-model-128-lego-diffuse-1000",
    },

    "video": {
        "svd": "stabilityai/stable-video-diffusion-img2vid",
        "svdxt": "stabilityai/stable-video-diffusion-img2vid-xt",
    },

    "longvideo": {
        "fifo_svd": "stabilityai/stable-video-diffusion-img2vid",
        "fifo_svdxt": "stabilityai/stable-video-diffusion-img2vid-xt",
    },
}