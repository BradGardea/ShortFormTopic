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

        process = subprocess.Popen(
            command,
            cwd=r"D:\utils\ComfyUI_windows_portable\ComfyUI\Practical-RIFE",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # env=env,
        )

        if process.returncode == 0:
            print("Subprocess completed successfully.")
            return 0
        else:
            print(f"Subprocess failed with return code {process.returncode}.")
            return -1

    except Exception as e:
        print(f"An error occurred while executing the subprocess: {e}")



def interpolate_ai_video(input_path):
    # Define the transformation sequence
    transformation_sequence = [2, 2, 4, 4]  # The sequence of multipliers
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
        "--prompt", f"Using realisitic characters and scenery in the artstyle similar to Shakespear's 'Hamlet' play with {color} {part}",
        "--part-index", str(part_index),
        "--fps", "2",
        "--frames", "55",
        "--steps", "30",
        "--initial-frame-prompt", "In the style of realistic contemporary art" + {story_obj.get("prompt", {}).get("seed", "a magestic sceen unfolds")}
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

def generate_ai_video(story_obj, process_id, full_comfy_path=r"D:\utils\ComfyUI_windows_portable", gpu_ids=(0, 1)):
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
