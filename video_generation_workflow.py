import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import argparse



def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
        print("Succesfully loaded from main")
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config
        print("Succesfully loaded from utils")

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    # Create an argument parser to handle command-line args
    parser = argparse.ArgumentParser(description="Process some inputs for video generation.")
    
    # Add the necessary arguments as per the previous context
    parser.add_argument('--cuda-device', type=int, default=0, help='GPU device to use (e.g., 0 for first GPU)')
    parser.add_argument('--id', type=str, default="default", help='ID for the generation')
    parser.add_argument('--prompt', type=str, default="", help='Text prompt to be processed')
    parser.add_argument('--part-index', type=int, default=0, help='Part index to be processed')
    parser.add_argument('--fps', type=int, default=12, help='fps to set')
    parser.add_argument('--frames', type=int, default=121, help='number of frames')
    parser.add_argument('--steps', type=int, default=60, help='Number of steps to apply')

    # Parse the arguments
    args = parser.parse_args()

    # Now, use the parsed arguments in the main logic
    print(f"Using GPU: {args.cuda_device}, Generation ID: {args.id}, Processing prompt: {args.prompt} with fps: {args.fps}, frames: {args.frames}, steps: {args.steps}")

    if args.cuda_device is not None:
        torch.cuda.set_device(int(args.cuda_device))
        torch.cuda.set_device(int(args.cuda_device))
        print("Set cuda device to: {}".format(args.cuda_device))

    # Proceed with the rest of the method (the same as before)
    import_custom_nodes()
    sys.stdout.write("Running generation")
    with torch.inference_mode():
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_38 = cliploader.load_clip(
            clip_name="t5\\google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
            type="mochi",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=args.prompt,
            clip=get_value_at_index(cliploader_38, 0),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="", clip=get_value_at_index(cliploader_38, 0)
        )

        emptymochilatentvideo = NODE_CLASS_MAPPINGS["EmptyMochiLatentVideo"]()
        emptymochilatentvideo_21 = emptymochilatentvideo.generate(
            width=512, height=512, length=args.frames, batch_size=1
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_37 = unetloader.load_unet(
            unet_name="mochi\\mochi_preview_dit_fp8_e4m3fn.safetensors",
            weight_dtype="default",
        )

        mochivaeloader = NODE_CLASS_MAPPINGS["MochiVAELoader"]()
        mochivaeloader_43 = mochivaeloader.loadmodel(
            model_name="mochi_preview_vae_decoder_bf16.safetensors", precision="bf16"
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_49 = vaeloader.load_vae(
            vae_name="mochi_preview_vae_decoder_bf16.safetensors"
        )

        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        mochidecode = NODE_CLASS_MAPPINGS["MochiDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        sys.stdout.write("Running sampler")
        for q in range(1):
            ksampler_3 = ksampler.sample(
                seed=6335316662,
                steps=args.steps,
                cfg=6,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(unetloader_37, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptymochilatentvideo_21, 0),
            )
            sys.stdout.write("Running decoder")
            mochidecode_55 = mochidecode.decode(
                enable_vae_tiling=True,
                auto_tile_size=True,
                frame_batch_size=6,
                tile_sample_min_height=160,
                tile_sample_min_width=288,
                tile_overlap_factor_height=0.1666,
                tile_overlap_factor_width=0.2,
                override_cpu_only=True,
                unnormalize=False,
                vae=get_value_at_index(mochivaeloader_43, 0),
                samples=get_value_at_index(ksampler_3, 0),
            )

            # saveimage_52 = saveimage.save_images(
            #     filename_prefix="frame_", images=get_value_at_index(mochidecode_55, 0)
            # )

            vhs_videocombine_54 = vhs_videocombine.combine_video(
                frame_rate=args.fps,
                loop_count=0,
                filename_prefix= os.path.join(str(args.id), str(args.part_index)),
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(mochidecode_55, 0),
                unique_id=13596902244891677597,
            )
            exit(0)

if __name__ == "__main__":
    main()
