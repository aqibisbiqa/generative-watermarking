

print(
    "\n".join([\
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