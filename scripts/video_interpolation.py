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


def execute_inference(input_video, output_video):
    try:
        command = [
            r"D:\utils\ComfyUI_windows_portable\python_embeded\python.exe",
            "inference_video.py",
            "--multi=4",
            f"--video={input_video}",
            f"--output={output_video}",
            f"--no_audio"
        ]

        # # Set the PYTHONPATH to include the script's directory
        # env = os.environ.copy()
        # env["PATH"] = r"D:\utils\ComfyUI_windows_portable\ComfyUI\Practical-RIFE"

        # Execute the subprocess
        process = subprocess.Popen(
            command,
            cwd=r"D:\utils\ComfyUI_windows_portable\ComfyUI\Practical-RIFE",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # env=env,
        )


        # Stream output and errors
        # for line in process.stdout:
        #     print(f"STDOUT: {line.strip()}")
        # for line in process.stderr:
        #     print(f"STDERR: {line.strip()}")

        # # Wait for the process to complete
        # process.wait()

        # output, err = process.communicate()

        # print(output)

        if process.returncode == 0:
            print("Subprocess completed successfully.")
        else:
            print(f"Subprocess failed with return code {process.returncode}.")

    except Exception as e:
        print(f"An error occurred while executing the subprocess: {e}")



def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    # User-defined variables
    file_name = "generated_video_00009"
    input_video_path = fr"D:\utils\ComfyUI_windows_portable\ComfyUI\playground\{file_name}.mp4"  # Path to the input video
    output_video_path = fr"D:\utils\ComfyUI_windows_portable\ComfyUI\playground\{file_name}_SUBPROCESS_video.mp4"  # Path to save the output video

    execute_inference(input_video_path, output_video_path)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")



if __name__ == '__main__':
    main()