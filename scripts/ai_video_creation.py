import os
import subprocess
from multiprocessing import Pool
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import logging
import cv2
import numpy as np
import torch
from torch import nn
import subprocess

from diffusers import (
    StableDiffusionPipeline,
    StableVideoDiffusionPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    I2VGenXLPipeline,
    CogVideoXImageToVideoPipeline,
    AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXPipeline
)
from diffusers.utils import load_image, export_to_video
from PIL import Image
import torch
import os
import gc

from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight

def generate_ai_video_stable_diffusion(story_obj, process_id, seed_image_path=None, video_fps=2, num_frames=14):
 
    parts_obj = story_obj.get("prompt", {}).get("parts", {})
    prompts = [
        {
            "seed": parts_obj.get(f"part{i}", {}).get("seed", ""),
            "motion": parts_obj.get(f"part{i}", {}).get("motion", "")
        }
        for i in range(1, len(parts_obj) + 1)
    ]
    
    if not prompts or not prompts[0]["seed"].strip():
        prompts = [
            {"seed": "Astronaut in a jungle, cold color palette, muted colors, detailed, realistic, 8k", "motion": "Astronaut walking through dense jungle with mist and glowing plants"},
            {"seed": "Astronaut exploring an underwater city, bioluminescent lights, futuristic, realistic, 8k", "motion": "Astronaut swimming through glowing coral and fish in an underwater city"},
            {"seed": "Astronaut on a futuristic desert planet, surreal colors, artistic, realistic, 8k", "motion": "Astronaut walking on a surreal desert with glowing sands and strange structures"}
        ]

    seed_prompt = story_obj.get("prompt", {}).get("seed", "Astronaut riding a horse, pale colors, detailed, realistic 8k") + " very realistic 8k."
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    output_folder = f"data/out/{process_id}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the initial pipeline for generating the seed image
    # textandimage_pipeline = AutoPipelineForText2Image.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    #     use_safetensors=True,
    #     token=""
    # ).to("cuda:0")
    textandimage_pipeline = None

    # Generate the initial seed image if not provided
    if not seed_image_path:
        print("Generating initial seed image from: ", seed_prompt)
        seed_image = textandimage_pipeline(seed_prompt).images[0]
        seed_image_path = os.path.join(output_folder, "seed.png")
        seed_image.save(seed_image_path)
        print("Saved to: ", seed_image_path)
    else:
        print("Using provided seed image...")
        seed_image = Image.open(seed_image_path)


    # quantization = int8_weight_only

    # text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-5b", subfolder="text_encoder", torch_dtype=torch.bfloat16)
    # quantize_(text_encoder, quantization())

    # transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b", subfolder="transformer", torch_dtype=torch.bfloat16)
    # quantize_(transformer, quantization())

    # vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b", subfolder="vae", torch_dtype=torch.bfloat16)
    # quantize_(vae, quantization())

    video_pipeline = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
    # text_encoder=text_encoder,
    # transformer=transformer,
    # vae=vae,
    device_map="balanced",
    max_memory={0: "10GB", 1: "10GB"}
    )

    video_pipeline.vae.enable_slicing()
    video_pipeline.vae.enable_tiling()

    current_image = seed_image
    for idx, stage in enumerate(prompts):
        seed = stage["seed"]
        motion = stage["motion"]

        print(f"Processing stage {idx + 1}/{len(prompts)} with seed: {seed} and motion: {motion}")

        gc.collect()
        torch.cuda.empty_cache()

        # resized_image = current_image.resize((1024, 576))
                
        frames = video_pipeline(
        prompt=motion,
        # image=resized_image,
        width=720,
        height=480,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator().manual_seed(8888),
        ).frames[0]

        video_path = os.path.join(output_folder, f"video_{idx + 1}.mp4")
        export_to_video(frames, video_path, fps=video_fps)
        print(f"Video saved to {video_path}")

        last_frame = frames[-1]

        last_frame_path = os.path.join(output_folder, f"frame_{idx + 1}.png")
        last_frame.resize((1024, 1024)).save(last_frame_path)

        print(f"Resized last frame saved as an image at {last_frame_path}")

        gc.collect()
        torch.cuda.empty_cache()

        if idx < len(prompts) - 1:
            current_image = textandimage_pipeline(
                seed,
                # image=last_frame.resize((1024, 1024)),
                strength=0.8,
                guidance_scale=9.5
            ).images[0]

            new_seed_path = os.path.join(output_folder, f"new_seed_{idx + 1}.png")
            current_image.resize((1024, 576)).save(new_seed_path)

            print(f"Resized new frame saved as an image at {new_seed_path}")

    print("All parts processed successfully.")
    return 0



def execute_inference(input_video, output_video, multi=2):
    try:
        command = [
            r"D:\utils\ComfyUI_windows_portable\python_embeded\python.exe",
            "inference_video.py",
            f"--multi={multi}",
            f"--video={input_video}",
            f"--output={output_video}",
            f"--no_audio"
        ]
        print("Running")

        process = subprocess.Popen(
            command,
            cwd=r"D:\utils\ComfyUI_windows_portable\ComfyUI\Practical-RIFE",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # env=env,
        )
        output, err = process.communicate()

        print(output)


        if process.returncode == 0:
            return 0
        else:
            return -1


    except Exception as e:
        print(f"An error occurred while executing the subprocess: {e}")

def interpolate_ai_video(input_path):
    # Define the transformation sequence
    transformation_sequence = [2, 4, 4]  # The sequence of multipliers
    initial_fps = 2
    current_fps = initial_fps

    # Construct paths
    input_video_path = os.path.abspath(input_path)
    input_dir = os.path.dirname(input_video_path)
    interpolized_dir = os.path.join(input_dir, "interpolized")
    os.makedirs(interpolized_dir, exist_ok=True)

    # Extract the base name of the input file
    input_file_name = os.path.basename(input_video_path)
    base_name, ext = os.path.splitext(input_file_name)

    # Track intermediate files for cleanup
    intermediate_files = []

    # Apply the interpolation step-by-step
    for idx, multi in enumerate(transformation_sequence):
        # Update the FPS
        current_fps *= multi

        # Construct the output file name for this step
        output_file_name = f"{base_name}_intermediate_{idx+1}_{current_fps}fps{ext}"
        output_video_path = os.path.join(interpolized_dir, output_file_name)

        # Perform the interpolation
        if execute_inference(input_video_path, output_video_path, multi) < 0:
            print("Error interploating video: ", input_video_path)
            return -1

        # Add the output to intermediate files for cleanup
        intermediate_files.append(output_video_path)

        # Set the output of this step as the input for the next step
        input_video_path = output_video_path

    # Rename the final output file
    final_output_file_name = f"{base_name}_interpolated_{current_fps}fps{ext}"
    final_output_path = os.path.join(interpolized_dir, final_output_file_name)
    os.rename(output_video_path, final_output_path)

    # Delete intermediate files
    for temp_file in intermediate_files:
        if os.path.exists(temp_file) and temp_file != final_output_path:
            os.remove(temp_file)

    print(f"Interpolation completed. Final video saved to: {final_output_path}")

def run_process(story_obj, process_id, part_index, full_comfy_path, gpu_id):
    python_path = os.path.join(full_comfy_path, "python_embeded", "python.exe")
    comfy_path = os.path.join(full_comfy_path, "ComfyUI")
    script_path = os.path.join(comfy_path, "video_generation_workflow.py")
    
    # Ensure the script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at {script_path}")
    
    # Get the prompt parts
    prompt_parts = [story_obj.get("prompt", {}).get("parts", {}).get(f"part{i}", "") for i in range(1,7)]
    style = story_obj.get("prompt", {}).get("style", "A BBC telenovela")
    color = story_obj.get("prompt", {}).get("color", "Soft, dreamy hues with flashes of vibrant colors in Wonderland to represent wonder and discovery.")
    if not prompt_parts or len(prompt_parts) < 6:
        raise ValueError("No or not enough parts found in the prompt to process.")
    part = prompt_parts[part_index]

    cmd = [
        python_path, script_path,
        "--cuda-device", str(gpu_id),
        "--id", str(process_id),
        "--prompt", f"In the style of very realistic contemporary art film with detailed characters and faces, {part}",
        "--part-index", str(part_index),
        "--fps", "2",
        "--frames", "61",
        "--steps", "60",
        # "--initial-frame-prompt", "In the style of realistic contemporary art" + str(story_obj.get("prompt", {}).get("seed", "a magestic sceen unfolds"))
    ]   
    try:
        process = subprocess.Popen(cmd, cwd=comfy_path, text=True)
        
        while True:
            out = process.poll()
            if out is None:
                continue
            else:
                print(f"GPU {gpu_id} process done with code: ", process.returncode)
                if process.returncode == 0:
                    return 0
                else:
                    return -1

    except Exception as e:
        print(f"Error running process for part {part_index}: {e}")
        return -1  # Failure return value

def generate_ai_video_mochi(story_obj, process_id, full_comfy_path=r"D:\utils\ComfyUI_windows_portable", gpu_ids=(0, 1)):
    """
    Generate and upscale video using a script, running processes in parallel for prompt parts.
    
    Parameters:
    - story_obj: The object containing information about the story.
    - comfy_path: Path to the working directory where the script resides.
    - gpu_ids: A tuple specifying GPU IDs for parallel processes.
    - process_id: An identifier for the processes for logging and debugging purposes.
    """
    prompt_parts = story_obj.get("prompt", {}).get("parts", [])
    # Process parts in pairs
    total_parts = len(prompt_parts)
    current_index = 0

    while current_index < total_parts:
        with Pool(processes=2) as pool:  # Limit to 2 concurrent processes
            batch_processes = []

            # Submit tasks to the pool for the current batch
            for gpu_id in gpu_ids:
                if current_index < total_parts:
                    part_index = current_index
                    batch_processes.append(
                        pool.apply_async(run_process, args=(story_obj, process_id, current_index, full_comfy_path, gpu_id))
                    )
                    current_index += 1

            pool.close()  # No more tasks will be added
            pool.join()   # Wait for all tasks in the current batch to finish

            # Check the results of each process in the batch
            for process in batch_processes:
                try:
                    result = process.get()
                    if result == 0:
                        print(f"Task for part {part_index} completed successfully.")
                    else:
                        print(f"Task for part {part_index} failed.")
                        return -1
                except Exception as e:
                    print(f"Error in task for part {part_index}: {e}")
                    return -1

    print("All parts processed successfully.")
    return 0



story_obj = {
    "prompt": {
        "seed": "A serene landscape of a futuristic city at sunrise",
        "parts": {
            "part1": {
                "seed": "Futuristic city skyline with tall glass buildings, warm hues, detailed, realistic, 8k",
                "motion": "Birds flying gracefully through the glowing skyline with the sun rising in the background"
            },
            "part2": {
                "seed": "A bustling marketplace with vibrant colors, detailed, realistic, 8k",
                "motion": "People walking through the crowded marketplace, vendors calling out, vibrant goods on display"
            },
            "part3": {
                "seed": "A tranquil forest with glowing mushrooms and serene ambience, detailed, realistic, 8k",
                "motion": "Misty forest with glowing mushrooms pulsating light, trees swaying gently in the wind"
            }
        }
    }
}


process_id = "example_project"
seed_image_path = r"D:\Brad\Projects\ShortFormSucker\data\out\example_project\seed.png" # Optional: Provide a path if you have a specific seed image
video_fps = 4  # Frames per second for the generated videos
num_frames = 40  # Number of frames per video

# Call the function
generate_ai_video_stable_diffusion(
    story_obj=story_obj,
    process_id=process_id,
    seed_image_path=seed_image_path,
    video_fps=video_fps,
    num_frames=num_frames
)

