import os
import json
import torch
import random
import numpy as np

COLORS = {
    'brown': [165, 42, 42],
    'red': [255, 0, 0],
    'pink': [253, 108, 158],
    'orange': [255, 165, 0],
    'yellow': [255, 255, 0],
    'purple': [128, 0, 128],
    'green': [0, 128, 0],
    'blue': [0, 0, 255],
    'white': [255, 255, 255],
    'gray': [128, 128, 128],
    'black': [0, 0, 0],
}

def hex_to_rgb(hex_string, return_nearest_color=False):
    r"""
    Covert Hex triplet to RGB triplet.
    """
    # Remove '#' symbol if present
    hex_string = hex_string.lstrip('#')
    # Convert hex values to integers
    red = int(hex_string[0:2], 16)
    green = int(hex_string[2:4], 16)
    blue = int(hex_string[4:6], 16)
    rgb = torch.FloatTensor((red, green, blue))[None, :, None, None]/255.
    if return_nearest_color:
        nearest_color = find_nearest_color(rgb)
        return rgb.cuda(), nearest_color
    return rgb.cuda()


def find_nearest_color(rgb):
    r"""
    Find the nearest neighbor color given the RGB value.
    """
    if isinstance(rgb, list) or isinstance(rgb, tuple):
        rgb = torch.FloatTensor(rgb)[None, :, None, None]/255.
    color_distance = torch.FloatTensor([np.linalg.norm(
        rgb - torch.FloatTensor(COLORS[color])[None, :, None, None]/255.) for color in COLORS.keys()])
    nearest_color = list(COLORS.keys())[torch.argmin(color_distance).item()]
    return nearest_color


def parse_color_input(raw_gen_boxes):
    """
    Given prompt and gen_boxes with colors
    output
    
    base_text_prompt, color_text_prompts, color_names, color_rgbs, use_grad_guidance
    """
    color_text_prompts = []
    color_rgbs = []
    color_names = []
    
    for box in raw_gen_boxes:
        color_rgb, nearest_color = hex_to_rgb(box[1], True)
        color_rgbs.append(color_rgb)
        color_names.append(nearest_color)
        color_text_prompts.append(box[0])
        
    return color_text_prompts, color_names, color_rgbs

def get_region_diffusion_input(model, base_text_prompt, color_text_prompts, color_names):
    r"""
    Algorithm 1 in the paper.
    """
    region_text_prompts = []
    region_target_token_ids = []
    base_tokens = model.tokenizer._tokenize(base_text_prompt)


    # process the color text prompt
    for color_text_prompt, color_name in zip(color_text_prompts, color_names):
        region_target_token_ids.append([])
        region_text_prompts.append(color_name+' '+color_text_prompt)
        style_tokens = model.tokenizer._tokenize(color_text_prompt)
        for style_token in style_tokens:
            region_target_token_ids[-1].append(
                base_tokens.index(style_token)+1)

    # process the remaining tokens without any attributes
    region_text_prompts.append(base_text_prompt)
    region_target_token_ids_all = [
        id for ids in region_target_token_ids for id in ids]
    target_token_ids_rest = [id for id in range(
        1, len(base_tokens)+1) if id not in region_target_token_ids_all]
    region_target_token_ids.append(target_token_ids_rest)

    region_target_token_ids = [torch.LongTensor(
        obj_token_id) for obj_token_id in region_target_token_ids]
    return region_text_prompts, region_target_token_ids, base_tokens


## This is important in terms of getting the tokens of target colors represented by tokens
def get_gradient_guidance_input(model, base_tokens, color_text_prompts, color_rgbs, 
                                guidance_start_step=999, color_guidance_weight=1):
    r"""
    Control the token impact using font sizes.
    """
    color_target_token_ids = []
    for text_prompt in color_text_prompts:
        color_target_token_ids.append([])
        color_tokens = model.tokenizer._tokenize(text_prompt)
        for color_token in color_tokens:
            color_target_token_ids[-1].append(base_tokens.index(color_token)+1)
    color_target_token_ids_all = [
        id for ids in color_target_token_ids for id in ids]
    color_target_token_ids_rest = [id for id in range(
        1, len(base_tokens)+1) if id not in color_target_token_ids_all]
    color_target_token_ids.append(color_target_token_ids_rest)
    color_target_token_ids = [torch.LongTensor(
        obj_token_id) for obj_token_id in color_target_token_ids]
    
    text_format_dict = {'word_pos': None, 'font_size': None}

    text_format_dict['target_RGB'] = color_rgbs
    text_format_dict['guidance_start_step'] = guidance_start_step
    text_format_dict['color_guidance_weight'] = color_guidance_weight
    return text_format_dict, color_target_token_ids

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
