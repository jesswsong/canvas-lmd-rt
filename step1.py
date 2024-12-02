import os
from prompt import get_prompts, prompt_types, template_versions
from utils import parse
from utils.parse import parse_input_with_negative, parse_input_from_canvas, bg_prompt_text, neg_prompt_text, filter_boxes, show_boxes
from utils.llm import get_llm_kwargs, get_full_prompt, get_layout, model_names
from utils import cache
import matplotlib.pyplot as plt
import argparse
import time

# This only applies to visualization in this file.
scale_boxes = False
if scale_boxes:
    print("Scaling the bounding box to fit the scene")
else:
    print("Not scaling the bounding box to fit the scene")

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--prompt-type", choices=prompt_types, default="demo")
    parser.add_argument("--auto-query", action='store_true', help='Auto query using the API')
    parser.add_argument("--always-save", action='store_true', help='Always save the layout without confirming')
    parser.add_argument("--no-visualize", action='store_true', help='No visualizations')
    parser.add_argument("--visualize-cache-hit", action='store_true', help='Save boxes for cache hit')
    args = parser.parse_args()
    
    visualize_cache_hit = args.visualize_cache_hit


    # This is for visualizing bounding boxes
    parse.img_dir = f"img_generations/imgs_{args.prompt_type}"
    if not args.no_visualize:
        os.makedirs(parse.img_dir, exist_ok=True)

    # create cache directory
    cache.cache_path = f'cache/cache_{args.prompt_type.replace("lmd_", "")}temp.json'
    print(f"Cache: {cache.cache_path}")
    os.makedirs(os.path.dirname(cache.cache_path), exist_ok=True)
    cache.cache_format = "json"

    cache.init_cache()

    # Read canvas input
    parsed_input = parse_input_from_canvas('/mnt/hd1/jwsong/dsc180/canvas-lmd-rt/canvas_input/data.json')#parse_input_with_negative(text=resp, no_input=args.auto_query)
    
    if parsed_input is None:
        raise ValueError("Invalid input")
    raw_gen_boxes, bg_prompt, neg_prompt, prompt = parsed_input 
                
    # Load canvas input into bounding boxes
    gen_boxes = [{'name': box[0], 'color': box[1], 'bounding_box': box[2]} for box in raw_gen_boxes]
    gen_boxes = filter_boxes(gen_boxes, scale_boxes=scale_boxes)
    
    if not args.no_visualize:
        show_boxes(gen_boxes, bg_prompt=bg_prompt, neg_prompt=neg_prompt)
        plt.clf()
        print(f"Visualize masks at {parse.img_dir}")
    
    # Save cache of this round
    response = f"{raw_gen_boxes}\n{bg_prompt_text}{bg_prompt}\n{neg_prompt_text}{neg_prompt}"
    cache.add_cache(prompt, response)
