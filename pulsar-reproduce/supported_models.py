from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

def get_pipeline(model):

    models_by_type = {
        "latent": {
            "sd15": (StableDiffusionPipeline, "runwayml/stable-diffusion-v1-5"),
            "sd21": (StableDiffusionPipeline, "stabilityai/stable-diffusion-2-1-base"),
            "sd21real": (StableDiffusionPipeline, "friedrichor/stable-diffusion-2-1-realistic"),
        },
        "pixel": {
            "church": (DDIMPipeline, "google/ddpm-church-256"),
            "bedroom": (DDIMPipeline, "google/ddpm-bedroom-256"),
            "cat": (DDIMPipeline, "google/ddpm-cat-256"),
            "celeb": (DDIMPipeline, "google/ddpm-celebahq-256"),

            "butterflies": (DDIMPipeline, "dboshardy/ddim-butterflies-128"),
            "lego": (DDIMPipeline, "lukasHoel/ddim-model-128-lego-diffuse-1000"),
        }
        
    }

    model_is_supported = False

    for model_type in models_by_type:
        if model in models_by_type[model_type]:
            model_is_supported = True
            pipeline_cls, model_id_or_path = models_by_type[model_type][model]
        if model_is_supported:
            break
    if not model_is_supported:
        raise NotImplementedError(f"the {model} model is not yet supported")

    return pipeline_cls, model_id_or_path