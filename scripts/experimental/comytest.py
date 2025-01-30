import random
import logging
import time
import threading
from reddit import run_scrapers
from azure_synth import get_tts
from video_generator import create_combined_video_for_post
import pandas as pd
import os
import uuid
from llama_generation import generate_response
import json
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait


def generate_and_upscale_video(story_obj, process_id, full_comfy_path=r"D:\utils\ComfyUI_windows_portable", gpu_ids=(0, 1)):
    """
    Generate and upscale video using a script, running processes in parallel for prompt parts.
    
    Parameters:
    - story_obj: The object containing information about the story.
    - comfy_path: Path to the working directory where the script resides.
    - gpu_ids: A tuple specifying GPU IDs for parallel processes.
    - process_id: An identifier for the processes for logging and debugging purposes.
    """
    python_path = os.path.join(full_comfy_path, "python_embeded", "python.exe")
    comfy_path = os.path.join(full_comfy_path, "ComfyUI")
    script_path = os.path.join(comfy_path, "video_generation_workflow.py")
    
    # Ensure the script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at {script_path}")
    
    # Get the prompt parts
    prompt_parts = story_obj.get("prompt", {}).get("parts", [])
    style = story_obj.get("prompt", {}).get("style", "A BBC telenovela")
    color = story_obj.get("prompt", {}).get("color", "Soft, dreamy hues with flashes of vibrant colors in Wonderland to represent wonder and discovery.")
    if not prompt_parts or len(prompt_parts) < 6:
        raise ValueError("No or not enough parts found in the prompt to process.")
    
    # Process parts in pairs
    total_parts = len(prompt_parts)
    current_index = 0

    def run_process(part_index, gpu_id):
        part = prompt_parts[part_index]
        cmd = [
            python_path, script_path,
            "--cuda-device", str(gpu_id),
            "--id", str(process_id),
            "--prompt", f"In the style of {style} with colors described as {color} With clear characters and movement, create a scene where: {part}",
            "--part-index", str(part_index)
        ]   
        try:
            process = subprocess.Popen(cmd, cwd=comfy_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            while True:
                retcode = process.poll()

                if retcode is None:  # Process is still running
                    output = process.stdout.readline()
                    if output:
                        print(f"[GPU {gpu_id} Part {part_index}] {output.strip()}")
                    
                    error_output = process.stderr.readline()
                    if error_output:
                        print(f"[GPU {gpu_id} Part {part_index} Error] {error_output.strip()}")
                else:
                    break

            process.stdout.close()
            stderr = process.stderr.read()
            if stderr:
                print(f"[GPU {gpu_id} Part {part_index} Error] {stderr.strip()}")

            if retcode == 0:
                print(f"Process for part {part_index} on GPU {gpu_id} completed successfully.")
                return 0  
            else:
                print(f"Error for part {part_index} on GPU {gpu_id}.")
                return -1 

        except Exception as e:
            print(f"Error running process for part {part_index}: {e}")
            return -1  # Failure return value

    # Create a ThreadPoolExecutor to run processes concurrently (2 at a time)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        while current_index < total_parts:
            for gpu_id in gpu_ids:
                if current_index < total_parts:
                    part_index = current_index
                    # Submit the task to the executor without blocking
                    futures.append(executor.submit(run_process, part_index, gpu_id))
                    current_index += 1
            wait(futures)

        # Monitor the return values of each future
        for future in futures:
            result = future.result()  # This will block if necessary
            if result == 0:
                print("Task completed successfully.")
            else:
                print("Task failed.")
                return -1

    print("All parts processed successfully.")

def generate_single_test(story_obj, process_id, full_comfy_path=r"D:\utils\ComfyUI_windows_portable", gpu_ids=(0, 1)):
    
    python_path = os.path.join(full_comfy_path, "python_embeded", "python.exe")
    comfy_path = os.path.join(full_comfy_path, "ComfyUI")
    script_path = os.path.join(comfy_path, "video_generation_workflow.py")
    
    # Ensure the script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at {script_path}")
    
    # Get the prompt parts
    prompt_parts = story_obj.get("prompt", {}).get("parts", [])
    style = story_obj.get("prompt", {}).get("style", "A BBC telenovela")
    color = story_obj.get("prompt", {}).get("color", "Soft, dreamy hues with flashes of vibrant colors in Wonderland to represent wonder and discovery.")
    if not prompt_parts or len(prompt_parts) < 6:
        raise ValueError("No or not enough parts found in the prompt to process.")
    
    # Process parts in pairs
    total_parts = len(prompt_parts)
    current_index = 0

    part_index = 0
    gpu_id = 0
    
    part = prompt_parts[0]
    cmd = [
        python_path, script_path,
        "--cuda-device", str(1),
        "--id", str(process_id),
        "--prompt", f"In the style of {style} with colors described as {color} With clear characters and movement, create a scene where: {part}",
        "--part-index", str(part_index),
        "--fps", "2",
        "--frames", "13",
        "--steps", "1",
        # "--windows-standalone-build",
        # "--port", str(8000 + gpu_id),
    ] 

    try:
        process = subprocess.Popen(cmd, cwd=comfy_path, text=True)
        print("here")

        while True:
            out = process.poll()
            if out is None:
                continue
            else:
                print("Proces done with code: ", process.returncode)

        # for line in iter(process.stdout.readline, ""):
        #     print(line.strip())

    #     stdout, stderr = process.communicate()  # Wait for the process to complete
        
    #     # Print the real-time output
    #     if stdout:
    #         print(f"[GPU {0} Part {part_index}] {stdout.strip()}")
        
    #     # Capture any error output
    #     if stderr:
    #         print(f"[GPU {0} Part {part_index} Error] {stderr.strip()}")

    #     # Check the return code of the process
    #     if process.returncode == 0:
    #         print(f"Process for part {part_index} on GPU {0} completed successfully.")
    #         return 0  # Success return value
    #     else:
    #         print(f"Error for part {part_index} on GPU {0}.")
    #         return -1  # Failure return value

    except Exception as e:
        print(f"Error running process for part {part_index}: {e}")
        return -1  # Failure return value

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(os.path.dirname(abspath))
    os.chdir(dname)
    # main()
    directory = "data/stories"  # Replace with your directory path
    file_name = "98485939-8a8b-4ba7-825b-321a3eaa232f.json"  # Replace with your file name

    # Construct the full file path
    file_path = os.path.join(directory, file_name)

    # Read the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load the JSON data as a Python dictionary
            generate_single_test(data, "98485939-8a8b-4ba7-825b-321a3eaa232f")
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")