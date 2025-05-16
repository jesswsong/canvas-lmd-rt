

# LMD imports
import os
from prompt import get_prompts, prompt_types, template_versions
from utils import parse, vis, cache
from utils.parse import parse_input_with_negative, parse_input_from_canvas, bg_prompt_text, neg_prompt_text, filter_boxes, show_boxes
from utils.llm import get_llm_kwargs, get_full_prompt, get_layout, model_names
from utils import cache
import matplotlib.pyplot as plt
import argparse
from generate import generate_image
import models
import diffusers
from models import sam
import generation.sdxl_refinement as sdxl
from tqdm import tqdm
import bdb
import traceback
import time


# RT imports
import json
from PIL import Image
import numpy as np
from rt_models.region_diffusion import RegionDiffusion
from rt_models.region_diffusion_sdxl import RegionDiffusionXL
from utils.attention_utils import get_token_maps
from utils.richtext_utils import seed_everything, get_attention_control_input, get_gradient_guidance_input
from utils.rt_utils import *


# This only applies to visualization in this file.
scale_boxes = False
if scale_boxes:
    print("Scaling the bounding box to fit the scene")
else:
    print("Not scaling the bounding box to fit the scene")


def entry_point(args):
    our_models = ["lmd", "lmd_plus"]

    if args.run_model == "lmd_plus":
        models.sd_key = "gligen/diffusers-generation-text-box"
        models.sd_version = "sdv1.4"

    else:
        models.sd_key = "runwayml/stable-diffusion-v1-5"
        models.sd_version = "sdv1.5"

    print(f"Using SD: {models.sd_key}")
    models.model_dict = models.load_sd(
        key=models.sd_key,
        use_fp16=False,
        scheduler_cls=diffusers.schedulers.__dict__[
            args.scheduler] if args.scheduler else None,
    )

    if args.run_model in our_models:
        sam_model_dict = sam.load_sam()
        models.model_dict.update(sam_model_dict)

    if args.run_model == "lmd":
        import generation.lmd as generation
    elif args.run_model == "lmd_plus":
        import generation.lmd_plus as generation
    elif args.run_model == "sd":
        if not args.ignore_negative_prompt:
            print(
                "**You are running SD without `ignore_negative_prompt`. This means that it still uses part of the LLM output and is not a real SD baseline that takes only the prompt."
            )
        import generation.stable_diffusion_generate as generation
    else:
        raise ValueError(f"Unknown model type: {args.run_model}")

    # Sanity check: the version in the imported module should match the `run_model`
    version = generation.version
    assert version == args.run_model, f"{version} != {args.run_model}"
    run = generation.run

    # set visualizations to no-op in batch generation
    for k in vis.__dict__.keys():
        if k.startswith("visualize"):
            vis.__dict__[k] = lambda *args, **kwargs: None

    # clear the figure when plt.show is called
    plt.show = plt.clf

    prompt_type = args.prompt_type
    template_version = args.template_version

    model = "model_temp_holder"
    cache.cache_format = "json"
    cache.cache_path = f'cache/cache_{args.prompt_type.replace("lmd_", "")}{"_" + template_version if template_version != "v5" else ""}_{model}.json'
    print(f"Loading LLM responses from cache {cache.cache_path}")
    cache.init_cache(allow_nonexist=True)

    save_suffix = ("_" + args.save_suffix) if args.save_suffix else ""
    repeats = args.repeats
    seed_offset = args.seed_offset

    base_save_dir = f"img_generations/img_generations_template{args.template_version}_{version}_{prompt_type}{save_suffix}"

    if args.sdxl:
        base_save_dir += f"_sdxl_{args.sdxl_step_ratio}"

    run_kwargs = {}

    argnames = float_args + int_args + str_args

    for argname in argnames:
        argvalue = getattr(args, argname)
        if argvalue is not None:
            run_kwargs[argname] = argvalue
            print(f"**Setting {argname} to {argvalue}**")

    if args.no_center_or_align:
        run_kwargs["align_with_overall_bboxes"] = False
        run_kwargs["so_center_box"] = False

    scale_boxes_default = not args.no_scale_boxes_default
    is_notebook = False

    if args.force_run_ind is not None:
        run_ind = args.force_run_ind
        save_dir = f"{base_save_dir}/run{run_ind}"
    else:
        run_ind = 0
        while True:
            save_dir = f"{base_save_dir}/run{run_ind}"
            if not os.path.exists(save_dir):
                break
            run_ind += 1

    print(f"Save dir: {save_dir}")

    if args.sdxl:
        # Offload model saves GPU memory.
        sdxl.init(offload_model=True)

    LARGE_CONSTANT = 123456789
    LARGE_CONSTANT2 = 56789
    LARGE_CONSTANT3 = 6789
    LARGE_CONSTANT4 = 7890

    ind = 0
    if args.regenerate > 1:
        # Need to fix the ind
        assert args.skip_first_prompts == 0

    # full_prompt = args.full_prompt # this used to be prompt.get_prompts in LMD
#     parsed_input = parse_input_from_canvas(args.ui_input_loc)

    for regenerate_ind in range(args.regenerate):
        print("regenerate_ind:", regenerate_ind)
        cache.reset_cache_access()
        # for prompt_ind, prompt in enumerate(tqdm(prompts, desc=f"Run: {save_dir}")):
        # For `save_as_display`:
        save_ind = 0
        kwargs = {}
        # prompt = full_prompt.strip().rstrip(".")
        ind_override = kwargs.get("seed", None)
        scale_boxes = kwargs.get("scale_boxes", scale_boxes_default)

        # # Load from cache
        # resp = cache.get_cache(prompt)

        # if resp is None:
        #     print(f"Cache miss, skipping prompt: {prompt}")
        #     ind += 1
        #     continue

        print(f"***run: {run_ind}, scale_boxes: {scale_boxes}***")
        parse.img_dir = f"{save_dir}/{ind}"
        # Skip if image is already generared
        os.makedirs(parse.img_dir, exist_ok=True)
        vis.reset_save_ind()
        try:
            parsed_input = parse_input_from_canvas(args.ui_input_loc)
            if parsed_input is None:
                raise ValueError("Invalid input")
            raw_gen_boxes, bg_prompt, neg_prompt, prompt = parsed_input
            # Load canvas input into bounding boxes
            # gen_boxes = [{'name': box[0], 'bounding_box': box[2]} for box in raw_gen_boxes]
            # TODO: here, box[1] is the colors associated with each box. it's not used in LMD yet but will be useful in the future as we integrate color in rt

            gen_boxes = [(box[0], box[2]) for box in raw_gen_boxes]
            # this format: [('deer', [100, 100, 300, 300])]
            gen_boxes = filter_boxes(gen_boxes, scale_boxes=scale_boxes)

            print(gen_boxes)

            spec = {
                "prompt": prompt,
                "gen_boxes": gen_boxes,
                "bg_prompt": bg_prompt,
                "extra_neg_prompt": neg_prompt,
            }

            print("spec:", spec)

            show_boxes(
                gen_boxes,
                bg_prompt=bg_prompt,
                neg_prompt=neg_prompt,
                show=is_notebook,
            )
            if not is_notebook:
                plt.clf()

            original_ind_base = (
                ind_override + regenerate_ind * LARGE_CONSTANT2
                if ind_override is not None
                else ind
            )

            for repeat_ind in range(repeats):
                # This ensures different repeats have different seeds.
                ind_offset = repeat_ind * LARGE_CONSTANT3 + seed_offset

                if args.run_model in our_models:
                    # Our models load `extra_neg_prompt` from the spec
                    if args.no_synthetic_prompt:
                        # This is useful when the object relationships cannot be expressed only by bounding boxes.
                        output = run(
                            spec=spec,
                            bg_seed=original_ind_base + ind_offset,
                            fg_seed_start=ind + ind_offset + LARGE_CONSTANT,
                            overall_prompt_override=prompt,
                            **run_kwargs,
                        )
                    else:
                        # Uses synthetic prompt (handles negation and additional languages better)
                        output = run(
                            spec=spec,
                            bg_seed=original_ind_base + ind_offset,
                            fg_seed_start=ind + ind_offset + LARGE_CONSTANT,
                            **run_kwargs,
                        )
                elif args.run_model == "sd":
                    output = run(
                        prompt=prompt,
                        seed=original_ind_base + ind_offset,
                        extra_neg_prompt=neg_prompt,
                        **run_kwargs,
                    )

                plain_img = output.image
                self_attn_maps = output.self_attn_maps
                cross_attn_maps = output.cross_attn_maps 
                n_maps = output.n_maps

                if args.sdxl:
                    plain_img = sdxl.refine(image=plain_img, spec=spec, refine_seed=original_ind_base +
                                         ind_offset + LARGE_CONSTANT4, refinement_step_ratio=args.sdxl_step_ratio)

                print(f"OUTPUT TYPE: {type(output)}")

                # TODO: oinput output to rt
                vis.display(output, "img", repeat_ind,
                            save_ind_in_filename=False)

        except (KeyboardInterrupt, bdb.BdbQuit) as e:
                print(e)
                exit()
        except RuntimeError:
            print(
                "***RuntimeError: might run out of memory, skipping the current one***"
            )
            print(traceback.format_exc())
            time.sleep(10)
        except Exception as e:
            print(f"***Error: {e}***")
            print(traceback.format_exc())
            if args.no_continue_on_error:
                raise e

        # try placing LMD output in rich text as input
        
            
    DIM_SCALAR = 8
    seed = args.seed_offset
    plain_img = Image.fromarray(plain_img)
    
    
    if (args.run_model == 'SD') or (args.run_model == 'lmd'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RegionDiffusion(device)
    elif args.run_model == 'lmd_plus':
        model = RegionDiffusionXL(
            load_path="stabilityai/stable-diffusion-xl-base-1.0")
    else:
        raise NotImplementedError
    
    
    # else:
    #     model.reset_attention_maps()

    use_grad_guidance = True
    base_text_prompt = prompt
    color_text_prompts, color_names, color_rgbs = parse_color_input(raw_gen_boxes)
    
    # create control input for region diffusion
    region_text_prompts, region_target_token_ids, base_tokens = get_region_diffusion_input(
        model, base_text_prompt, color_text_prompts, color_names)

    # create control input for region guidance
    text_format_dict, color_target_token_ids = get_gradient_guidance_input(
        model, base_tokens, color_text_prompts, color_rgbs, color_guidance_weight=args.color_guidance_weight)

    # TODO: figure out how these variables are all implemented
    height, width = output.shape[0], output.shape[1]
    # print(model.selfattn_maps) #this is currently empty
    color_obj_masks = get_token_maps(self_attn_maps, cross_attn_maps, n_maps, args.run_dir,
                                    height//DIM_SCALAR, width//DIM_SCALAR, color_target_token_ids[:-1], seed,
                                    base_tokens, segment_threshold=args.segment_threshold, num_segments=args.num_segments)
    color_obj_atten_all = torch.zeros_like(color_obj_masks[-1])
    for obj_mask in color_obj_masks[:-1]:
        color_obj_atten_all += obj_mask
    color_obj_masks = [transforms.functional.resize(color_obj_mask, (height, width),
                                                    interpolation=transforms.InterpolationMode.BICUBIC,
                                                    antialias=True)
                    for color_obj_mask in color_obj_masks]
    text_format_dict['color_obj_atten'] = color_obj_masks
    text_format_dict['color_obj_atten_all'] = color_obj_atten_all

    seed_everything(seed)
    model.masks = get_token_maps(self_attn_maps, cross_attn_maps, n_maps, args.run_dir,
                                height//DIM_SCALAR, width//DIM_SCALAR, region_target_token_ids[:-1], seed,
                                base_tokens, segment_threshold=args.segment_threshold, num_segments=args.num_segments)
    model.remove_tokenmap_hooks()

    # generate image from rich text
    begin_time = time.time()
    seed_everything(seed)
    fn_style = os.path.join(args.run_dir, 'seed%d_rich.jpg' % (seed))
    if args.model == 'SD':
        # print(f"MODEL PROMPT_TO_IMG: {text_format_dict}")

        rich_img = model.prompt_to_img(region_text_prompts, [negative_prompt],
                                    height=height, width=width, num_inference_steps=args.color_sample_steps,
                                    guidance_scale=args.color_guidance_weight, use_guidance=use_grad_guidance,
                                    inject_selfattn=args.inject_selfattn, text_format_dict=text_format_dict,
                                    inject_background=args.inject_background)
        imageio.imwrite(fn_style, rich_img[0])
    else:
        # print(] for text_format_dict['target']])
        print(f"MODEL SAMPLE: {text_format_dict}")
        rich_img = model.sample(region_text_prompts, [negative_prompt],
                                    height=height, width=width, num_inference_steps=args.color_sample_steps,
                                    guidance_scale=args.color_guidance_weight, use_guidance=use_grad_guidance,
                                    inject_selfattn=args.inject_selfattn, text_format_dict=text_format_dict,
                                    inject_background=args.inject_background, run_rich_text=True)
        rich_img.images[0].save(fn_style)
    print('time lapses to generate image from rich text: %.4f' %
        (time.time()-begin_time))
        
    
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-suffix", default=None, type=str)
    parser.add_argument("--repeats", default=1, type=int, help="Number of samples for each prompt")
    parser.add_argument("--regenerate", default=1, type=int, help="Number of regenerations. Different from repeats, regeneration happens after everything is generated")
    parser.add_argument("--force_run_ind", default=None, type=int, help="If this is enabled, we use this run_ind and skips generated images. If this is not enabled, we create a new run after existing runs.")
    parser.add_argument("--skip_first_prompts", default=0, type=int, help="Skip the first prompts in generation (useful for parallel generation)")
    parser.add_argument("--seed_offset", default=6, type=int, help="Offset to the seed (seed starts from this number)")
    parser.add_argument("--num_prompts", default=None, type=int, help="The number of prompts to generate (useful for parallel generation)")
    parser.add_argument(
        "--run-model",
        default="lmd_plus",
        choices=[
            "lmd",
            "lmd_plus",
            "sd"
        ],
    )
    parser.add_argument("--scheduler", default=None, type=str)
    parser.add_argument("--no-synthetic-prompt", action="store_true", help="Use the original prompt for overall generation rather than a synthetic prompt ([background prompt] with [objects])")
    parser.add_argument("--no-scale-boxes-default", action="store_true", help="Do not scale the boxes to fill the scene")
    parser.add_argument("--no-center-or-align", action="store_true", help="Do not perform per-box generation in the center and then align for overall generation")
    parser.add_argument("--no-continue-on-error", action="store_true")
    parser.add_argument("--prompt-type", choices=prompt_types, default="lmd")
    parser.add_argument("--template_version", choices=template_versions, default="v0.1")

    parser.add_argument("--sdxl", action="store_true", help="Enable sdxl.")
    parser.add_argument("--sdxl-step-ratio", type=float, default=0.3, help="SDXL step ratio: the higher the stronger the refinement.")

    float_args = [
        "frozen_step_ratio",
        "loss_threshold",
        "ref_ca_loss_weight",
        "fg_top_p",
        "bg_top_p",
        "overall_fg_top_p",
        "overall_bg_top_p",
        "fg_weight",
        "bg_weight",
        "overall_fg_weight",
        "overall_bg_weight",
        "overall_loss_threshold",
        "fg_blending_ratio",
        "mask_th_for_point",
        "so_floor_padding",
    ]
    for float_arg in float_args:
        parser.add_argument("--" + float_arg, default=None, type=float)

    int_args = [
        "loss_scale",
        "max_iter",
        "max_index_step",
        "overall_max_iter",
        "overall_max_index_step",
        "overall_loss_scale",
        # Set to 0 to disable and set to 1 to enable
        "horizontal_shift_only",
        "so_horizontal_center_only",
        # Set to 0 to disable and set to 1 to enable (default: see the default value in each generation file):
        "use_autocast",
        # Set to 0 to disable and set to 1 to enable
        "use_ref_ca"
    ]
    for int_arg in int_args:
        parser.add_argument("--" + int_arg, default=None, type=int)
    str_args = ["so_vertical_placement"]
    for str_arg in str_args:
        parser.add_argument("--" + str_arg, default=None, type=str)
    parser.add_argument("--multidiffusion_bootstrapping", default=20, type=int)
    parser.add_argument('--ui_input_loc', default="/mnt/hd1/jwsong/dsc180/canvas_lmd_rt/canvas_input/data.json",type=str)
    parser.add_argument('--run_dir', type=str, default='results/')
    parser.add_argument('--color_guidance_weight', type=float, default=0.5)
    parser.add_argument('--segment_threshold', type=float, default=0.3)
    parser.add_argument('--num_segments', type=int, default=9)
    parser.add_argument('--color_sample_steps', type=int, default=41)
    parser.add_argument('--inject_selfattn', type=float, default=0.)
    parser.add_argument('--inject_background', type=float, default=0.)
    
    args = parser.parse_args()
    entry_point(args)
    
    
    