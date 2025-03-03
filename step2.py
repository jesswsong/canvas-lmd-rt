


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

# This only applies to visualization in this file.
scale_boxes = False
if scale_boxes:
    print("Scaling the bounding box to fit the scene")
else:
    print("Not scaling the bounding box to fit the scene")
    
    
def entry_point(args):
    our_models = ["lmd", "lmd_plus"]
    gligen_models = ["gligen", "lmd_plus"]
    
    if args.run_model in gligen_models:
        models.sd_key = "gligen/diffusers-generation-text-box"
        models.sd_version = "sdv1.4"
        
    else:
        models.sd_key = "runwayml/stable-diffusion-v1-5"
        models.sd_version = "sdv1.5"

    print(f"Using SD: {models.sd_key}")
    models.model_dict = models.load_sd(
        key=models.sd_key,
        use_fp16=False,
        scheduler_cls=diffusers.schedulers.__dict__[args.scheduler] if args.scheduler else None,
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
    elif args.run_model == "gligen":
        import generation.gligen as generation
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
            #TODO: here, box[1] is the colors associated with each box. it's not used in LMD yet but will be useful in the future as we integrate color in rt
            
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
                elif args.run_model == "multidiffusion":
                    output = run(
                        gen_boxes=gen_boxes,
                        bg_prompt=bg_prompt,
                        original_ind_base=original_ind_base + ind_offset,
                        bootstrapping=args.multidiffusion_bootstrapping,
                        extra_neg_prompt=neg_prompt,
                        **run_kwargs,
                    )
                elif args.run_model == "backward_guidance":
                    output = run(
                        spec=spec,
                        bg_seed=original_ind_base + ind_offset,
                        **run_kwargs,
                    )
                elif args.run_model == "boxdiff":
                    output = run(
                        spec=spec,
                        bg_seed=original_ind_base + ind_offset,
                        **run_kwargs,
                    )
                elif args.run_model == "gligen":
                    output = run(
                        spec=spec,
                        bg_seed=original_ind_base + ind_offset,
                        **run_kwargs,
                    )

                print(f"OUTPUT TYPE: {type(output)}")
                output = output.image
                print(f"OUTPUT TYPE: {output.shape}")

                if args.sdxl:
                    output = sdxl.refine(image=output, spec=spec, refine_seed=original_ind_base + ind_offset + LARGE_CONSTANT4, refinement_step_ratio=args.sdxl_step_ratio)

                print(f"OUTPUT TYPE: {type(output)}")
                
                # TODO: oinput output to rt
                vis.display(output, "img", repeat_ind, save_ind_in_filename=False)

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

            




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-suffix", default=None, type=str)
    parser.add_argument("--repeats", default=1, type=int, help="Number of samples for each prompt")
    parser.add_argument("--regenerate", default=1, type=int, help="Number of regenerations. Different from repeats, regeneration happens after everything is generated")
    parser.add_argument("--force_run_ind", default=None, type=int, help="If this is enabled, we use this run_ind and skips generated images. If this is not enabled, we create a new run after existing runs.")
    parser.add_argument("--skip_first_prompts", default=0, type=int, help="Skip the first prompts in generation (useful for parallel generation)")
    parser.add_argument("--seed_offset", default=0, type=int, help="Offset to the seed (seed starts from this number)")
    parser.add_argument("--num_prompts", default=None, type=int, help="The number of prompts to generate (useful for parallel generation)")
    parser.add_argument(
        "--run-model",
        default="lmd_plus",
        choices=[
            "lmd",
            "lmd_plus",
            "sd",
            "multidiffusion",
            "backward_guidance",
            "boxdiff",
            "gligen",
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
    parser.add_argument('--ui_input_loc', default="/mnt/hd1/jwsong/dsc180/canvas-lmd-rt/canvas_input/data.json",type=str)

    args = parser.parse_args()
    entry_point(args)

    
# def entry_point(args, param):
#     # visualize_cache_hit = args.visualize_cache_hi
#     template_version = args.template_version
    
#     # Visualize bounding boxes
#     parse.img_dir = f"img_generations/imgs_{args.prompt_type}_template{template_version}"
#     if not args.no_visualize:
#         os.makedirs(parse.img_dir, exist_ok=True)
        
#     # Create LMD cache directory
#     cache.cache_path = f'cache/cache_{args.prompt_type.replace("lmd_", "")}{"_" + template_version if args.template_version != "v5" else ""}.json'
#     print(f"Cache: {cache.cache_path}")
#     os.makedirs(os.path.dirname(cache.cache_path), exist_ok=True)
#     cache.cache_format = "json"
#     cache.init_cache()
    
#     """
#     Get Prompt
#     """
#     full_prompt = args.full_prompt # this used to be prompt.get_prompts in LMD
#     parsed_input = parse_input_from_canvas(args.ui_input_loc)
    
#     if parsed_input is None:
#         raise ValueError("Invalid input")
#     raw_gen_boxes, bg_prompt, neg_prompt = parsed_input 


#     """
#     LMD processing
#     """
#     # Load canvas input into bounding boxes
#     gen_boxes = [{'name': box[0], 'bounding_box': box[1]} for box in raw_gen_boxes]
#     gen_boxes = filter_boxes(gen_boxes, scale_boxes=scale_boxes)
    
#     spec = {
#         "prompt": full_prompt,
#         "gen_boxes": gen_boxes,
#         "bg_prompt": bg_prompt,
#         "extra_neg_prompt": neg_prompt,
#     }

#     if not args.no_visualize:
#         show_boxes(gen_boxes, bg_prompt=bg_prompt, neg_prompt=neg_prompt)
#         plt.clf()
#         print(f"Visualize masks at {parse.img_dir}")
    
#     # Save cache of this round
#     response = f"{raw_gen_boxes}\n{bg_prompt_text}{bg_prompt}\n{neg_prompt_text}{neg_prompt}"
#     cache.add_cache(full_prompt, response)

#     # TODO：args.run_model
#     # imported from line 130 of generate.py
#     if args.run_model == "lmd":
#         import generation.lmd as generation
#     elif args.run_model == "lmd_plus":
#         import generation.lmd_plus as generation
#     elif args.run_model == "sd":
#         if not args.ignore_negative_prompt:
#             print(
#                 "**You are running SD without `ignore_negative_prompt`. This means that it still uses part of the LLM output and is not a real SD baseline that takes only the prompt."
#             )
#         import generation.stable_diffusion_generate as generation
#     elif args.run_model == "multidiffusion":
#         import generation.multidiffusion as generation
#     elif args.run_model == "backward_guidance":
#         import generation.backward_guidance as generation
#     elif args.run_model == "boxdiff":
#         import generation.boxdiff as generation
#     elif args.run_model == "gligen":
#         import generation.gligen as generation
#     else:
#         raise ValueError(f"Unknown model type: {args.run_model}")

#     # Sanity check: the version in the imported module should match the `run_model`
#     version = generation.version
#     assert version == args.run_model, f"{version} != {args.run_model}"
#     run = generation.run
#     if args.use_sdv2:
#         version = f"{version}_sdv2"
    
#     # TODO: let LMD generate its plain image
#     # from line 383 of generate.py
#     prompt = full_prompt.strip().rstrip(".")
#     output = run(prompt=prompt,
#                  seed=original_ind_base + ind_offset,
#                  extra_neg_prompt=neg_prompt,
#                  **run_kwargs,
#     )
#     output = output.image
#     vis.display(output, "img", repeat_ind, save_ind_in_filename=False)

    
#     """
#     Run Generate
#     """
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--save-suffix", default=None, type=str)
#     # parser.add_argument("--model", choices=model_names, required=True, help="LLM model to load the cache from")
#     # parser.add_argument("--repeats", default=1, type=int, help="Number of samples for each prompt")
#     # parser.add_argument("--regenerate", default=1, type=int, help="Number of regenerations. Different from repeats, regeneration happens after everything is generated")
#     # parser.add_argument("--force_run_ind", default=None, type=int, help="If this is enabled, we use this run_ind and skips generated images. If this is not enabled, we create a new run after existing runs.")
#     # parser.add_argument("--skip_first_prompts", default=0, type=int, help="Skip the first prompts in generation (useful for parallel generation)")
#     # parser.add_argument("--seed_offset", default=0, type=int, help="Offset to the seed (seed starts from this number)")
#     # parser.add_argument("--num_prompts", default=None, type=int, help="The number of prompts to generate (useful for parallel generation)")
#     # parser.add_argument(
#     #     "--run-model",
#     #     default="lmd_plus",
#     #     choices=[
#     #         "lmd",
#     #         "lmd_plus",
#     #         "sd",
#     #         "multidiffusion",
#     #         "backward_guidance",
#     #         "boxdiff",
#     #         "gligen",
#     #     ],
#     # )
#     # parser.add_argument("--scheduler", default=None, type=str)
#     # parser.add_argument("--use-sdv2", action="store_true")
#     # parser.add_argument("--ignore-bg-prompt", action="store_true", help="Ignore the background prompt (set background prompt to an empty str)")
#     # parser.add_argument("--ignore-negative-prompt", action="store_true", help="Ignore the additional negative prompt generated by LLM")
#     # parser.add_argument("--no-synthetic-prompt", action="store_true", help="Use the original prompt for overall generation rather than a synthetic prompt ([background prompt] with [objects])")
#     # parser.add_argument("--no-scale-boxes-default", action="store_true", help="Do not scale the boxes to fill the scene")
#     # parser.add_argument("--no-center-or-align", action="store_true", help="Do not perform per-box generation in the center and then align for overall generation")
#     # parser.add_argument("--no-continue-on-error", action="store_true")
#     # parser.add_argument("--prompt-type", choices=prompt_types, default="lmd")
#     # parser.add_argument("--template_version", choices=template_versions, required=True)
#     # parser.add_argument("--dry-run", action="store_true", help="skip the generation")

#     # parser.add_argument("--sdxl", action="store_true", help="Enable sdxl.")
#     # parser.add_argument("--sdxl-step-ratio", type=float, default=0.3, help="SDXL step ratio: the higher the stronger the refinement.")

#     # float_args = [
#     #     "frozen_step_ratio",
#     #     "loss_threshold",
#     #     "ref_ca_loss_weight",
#     #     "fg_top_p",
#     #     "bg_top_p",
#     #     "overall_fg_top_p",
#     #     "overall_bg_top_p",
#     #     "fg_weight",
#     #     "bg_weight",
#     #     "overall_fg_weight",
#     #     "overall_bg_weight",
#     #     "overall_loss_threshold",
#     #     "fg_blending_ratio",
#     #     "mask_th_for_point",
#     #     "so_floor_padding",
#     # ]
#     # for float_arg in float_args:
#     #     parser.add_argument("--" + float_arg, default=None, type=float)

#     # int_args = [
#     #     "loss_scale",
#     #     "max_iter",
#     #     "max_index_step",
#     #     "overall_max_iter",
#     #     "overall_max_index_step",
#     #     "overall_loss_scale",
#     #     # Set to 0 to disable and set to 1 to enable
#     #     "horizontal_shift_only",
#     #     "so_horizontal_center_only",
#     #     # Set to 0 to disable and set to 1 to enable (default: see the default value in each generation file):
#     #     "use_autocast",
#     #     # Set to 0 to disable and set to 1 to enable
#     #     "use_ref_ca"
#     # ]
#     # for int_arg in int_args:
#     #     parser.add_argument("--" + int_arg, default=None, type=int)
#     # str_args = ["so_vertical_placement"]
#     # for str_arg in str_args:
#     #     parser.add_argument("--" + str_arg, default=None, type=str)
#     # parser.add_argument("--multidiffusion_bootstrapping", default=20, type=int)

#     # args = parser.parse_args()
#     # generate_image(args)
    
        

# if __name__ == '__main__':
#     parser1 = argparse.ArgumentParser()
    
#     # parser.add_argument('--run_dir', type=str, default='results/')
#     parser1.add_argument('--height', type=int, default=None)
#     parser1.add_argument('--width', type=int, default=None)
#     parser1.add_argument('--seed', type=int, default=6)
#     parser1.add_argument('--sample_steps', type=int, default=41)
#     # parser.add_argument('--rich_text_json', type=str,
#     #                     default='{"ops":[{"insert":"A close-up 4k dslr photo of a "},{"attributes":{"link":"A cat wearing sunglasses and a bandana around its neck."},"insert":"cat"},{"insert":" riding a scooter. There are palm trees in the background."}]}')
#     parser1.add_argument('--negative_prompt', type=str, default='')
#     # parser.add_argument('--model', type=str, default='SD', choices=['SD', 'SDXL'])
#     parser1.add_argument('--guidance_weight', type=float, default=8.5)
#     # parser.add_argument('--color_guidance_weight', type=float, default=0.5)
#     # parser.add_argument('--inject_selfattn', type=float, default=0.)
#     # parser.add_argument('--segment_threshold', type=float, default=0.3)
#     # parser.add_argument('--num_segments', type=int, default=9)
#     # parser.add_argument('--inject_background', type=float, default=0.)
    
#     # lmd args
#     parser1.add_argument("--prompt-type", choices=prompt_types, default="demo")
#     parser1.add_argument("--model", choices=model_names, required=True)
#     parser1.add_argument("--template_version", choices=template_versions, required=True)
#     parser1.add_argument("--auto-query", action='store_true', help='Auto query using the API')
#     parser1.add_argument("--always-save", action='store_true', help='Always save the layout without confirming')
#     parser1.add_argument("--no-visualize", action='store_true', help='No visualizations')
#     parser1.add_argument("--visualize-cache-hit", action='store_true', help='Save boxes for cache hit')
    
#     parser1.add_argument("--ui-input-loc", type=str, required=True, help="Path to the input JSON file.")
#     parser1.add_argument("--full-prompt", type=str, required=True, help="Full prompt string to pass.")
    
    

#     args1 = parser1.parse_args()
#     default_resolution = 512 if args1.model == 'SD' else 1024
    
#     rich_text_json_temp = '{"ops":[{"insert":"a Gothic "},{"attributes":{"color":"#fd6c9e"},"insert":"church"},{"insert":" in a sunset with a beautiful landscape in the background."}]}'
#     print(rich_text_json_temp)
#     param = {
#         'text_input': json.loads(rich_text_json_temp),
#         'height': args1.height if args1.height is not None else default_resolution,
#         'width': args1.width if args1.width is not None else default_resolution,
#         'guidance_weight': args1.guidance_weight,
#         'steps': args1.sample_steps,
#         'noise_index': args1.seed,
#         'negative_prompt': args1.negative_prompt,
#     }

#     entry_point(args1, param)
