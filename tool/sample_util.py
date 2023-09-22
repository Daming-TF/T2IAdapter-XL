import torch
import numpy as np
import random
import tqdm
import copy
from basicsr.utils import tensor2img


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def inference(
        tokenizers, scheduler, model, vae, text_encoders, device, weight_dtype,
        prompt, size, prompt_n='', adapter_features=None, guidance_scale=7.5, seed=-1, steps=50):
    prompt_batch = [prompt_n, prompt]
    prompt_embeds, unet_added_cond_kwargs = compute_embeddings(device,
        prompt_batch=prompt_batch, proportion_empty_prompts=0, text_encoders=text_encoders,
        tokenizers=tokenizers, size=size
    )

    scheduler.set_timesteps(steps)

    if seed != -1:
        seed_everything(seed)
    noisy_latents = torch.randn((1, 4, size[0] // 8, size[1] // 8)).to("cuda").to(dtype=weight_dtype)

    with torch.no_grad():
        for t in tqdm.tqdm(scheduler.timesteps):
            with torch.no_grad():
                input = torch.cat([noisy_latents] * 2)

                noise_pred = model(
                    input,
                    t,
                    encoder_hidden_states=prompt_embeds["prompt_embeds"],
                    added_cond_kwargs=unet_added_cond_kwargs,
                    down_block_additional_residuals=copy.deepcopy(adapter_features),
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noisy_latents = scheduler.step(noise_pred, t, noisy_latents)[0]

    noisy_latents = noisy_latents.to(dtype=vae.dtype)
    # image = vae.decode(noisy_latents.cpu() / vae.config.scaling_factor, return_dict=False)[0]
    image = vae.decode(noisy_latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = tensor2img(image, rgb2bgr=False)

    return image


def encode_prompt(tokenizers, text_encoders, prompt_batch, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def compute_embeddings(device, prompt_batch, proportion_empty_prompts, text_encoders, tokenizers, size, is_train=True):
    original_size = size
    target_size = size
    crops_coords_top_left = (0, 0)

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        tokenizers, text_encoders, prompt_batch, proportion_empty_prompts, is_train
    )
    add_text_embeds = pooled_prompt_embeds

    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
    add_time_ids = add_time_ids.to(device, dtype=prompt_embeds.dtype)
    unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    return {"prompt_embeds": prompt_embeds}, unet_added_cond_kwargs