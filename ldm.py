import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, EulerDiscreteScheduler
import pathlib as p

if __name__=="__main__":

    # Create texture
    model_id = "stabilityai/stable-diffusion-2-1-base"
    device = "cuda"

    # lms = LMSDiscreteScheduler(
    #     beta_start=0.00085, 
    #     beta_end=0.012, 
    #     beta_schedule="scaled_linear"
    # )
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(model_id,scheduler=scheduler, use_auth_token=True, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    prompt = "a photo of an astronaut riding a horse on mars"
    images = []
    sample_num = 15

    for i in range(sample_num):
        with autocast("cuda"):
            '''
            output_dict
            ['sample'] - [PIL Image]
            ['nsfw_content_detected'] - [bool]
            '''
            output_dict = pipe(prompt, width=256,height=256,guidance_scale=7.5,prompt_strength=0.8,num_inference_steps=100)
            if output_dict["nsfw_content_detected"][0]:
                continue
            image = output_dict["sample"][0]
            text_impath = f"img_{i}.png"
            text_path = str(p.Path.cwd() / 'out' / text_impath)
            image.save(text_path)