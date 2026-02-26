from moviepy.editor import *
from moviepy.config import change_settings
from scipy.ndimage import gaussian_filter
import random
import os
import logging
from datetime import timedelta
from PIL import Image
import numpy as np
import math
from moviepy.video.tools.segmenting import findObjects
import pysrt
import re
import uuid
import datetime
import tqdm
import traceback
from moviepy.video.fx.resize import resize




FPS = 90

#region old

def video_has_too_many_black_pixels(video_clip, max_frames=10, threshold=20, moving_avg_window=5):
    """
    Check if a video has too many black pixels using a moving average.

    Parameters:
        video_clip (VideoFileClip): The video clip to analyze.
        max_frames (int): Number of frames to analyze.
        threshold (float): Percentage threshold of black pixels to reject the video.
        moving_avg_window (int): Window size for moving average.

    Returns:
        bool: True if the video has too many black pixels, otherwise False.
    """
    frame_count = 0
    black_pixel_percentages = []

    for frame in video_clip.iter_frames():
        if frame_count >= max_frames:
            break
        
        # Calculate the percentage of black pixels in the current frame
        black_pixel_percentage = calculate_black_pixel_percentage(frame)
        black_pixel_percentages.append(black_pixel_percentage)

        # Check moving average
        if len(black_pixel_percentages) >= moving_avg_window:
            moving_avg = np.mean(black_pixel_percentages[-moving_avg_window:])
            if moving_avg > threshold:
                return True  # Reject this video

        frame_count += 1

    return False

def select_and_trim_videos(duration_needed, video_folder="data/satisfying_videos", mode="default", videos=None):
    """ 
    Select videos based on mode, check for black pixel content, and trim or loop them to fit the needed duration.
    
    Modes:
    - default: Randomly select videos from 'loops' and 'regular' folders.
    - custom: Use the provided list of videos, loop each equally to fit the required duration.
    """
    # Handle 'custom' mode
    if mode == "custom":
        if not videos or len(videos) == 0:
            logging.error("Custom mode requires a non-empty list of videos.")
            return None
        
        # Calculate equal looping duration for each video
        num_videos = len(videos)
        duration_per_video = duration_needed / num_videos

        selected_videos = []
        for video_file in videos:
            try:
                video_clip = VideoFileClip(video_file)
                video_clip = loop_video_clip(video_clip, duration_per_video)
                selected_videos.append(video_clip)
            except Exception as e:
                logging.error(f"Error processing video '{video_file}': {e}")
                return None

        return selected_videos

    # Default behavior for 'default' mode
    # Get all video files from both 'loops' and 'regular' folders
    video_files = []
    for subfolder in ['loops', 'regular']:
        folder_path = os.path.join(video_folder, subfolder)
        if os.path.exists(folder_path):
            video_files.extend([os.path.join(folder_path, f) 
                                for f in os.listdir(folder_path) 
                                if f.endswith(('.mp4', '.mkv', '.mov'))])
    
    selected_videos = []
    total_duration = 0

    while total_duration < duration_needed and video_files:
        max_checks = 100
        curr_checks = 0
        video_clip = None

        while video_clip is None and curr_checks < max_checks:
            try:
                video_file = random.choice(video_files)
                video_clip = VideoFileClip(video_file)
                video_duration = video_clip.duration
                
                # Check for too many black pixels
                if video_has_too_many_black_pixels(video_clip):
                    logging.info(f"Video '{video_file}' rejected due to high black pixel content.")
                    video_clip = None
                    continue

                logging.info(f"Video '{video_file}' selected.")
            except Exception as e:
                logging.error(f"Error processing video '{video_file}': {e}")
                video_clip = None

            curr_checks += 1

        if video_clip is None:
            logging.error("Max attempts reached. Unable to find a valid video.")
            break

        # Check if the video is from the 'loops' folder and loop it if it's less than 20 seconds
        if 'loops' in video_file and video_clip.duration < 20:
            video_clip = loop_video_clip(video_clip, 20)

        if total_duration + video_clip.duration <= duration_needed:
            selected_videos.append(video_clip)
            total_duration += video_clip.duration
        else:
            # If adding this video exceeds the duration needed, trim it
            trimmed_video = video_clip.subclip(0, duration_needed - total_duration)
            selected_videos.append(trimmed_video)
            total_duration += trimmed_video.duration
            trimmed_video.close()
            break  # Stop once we've reached the required duration

    return selected_videos

#endregion


def blur(clip, sigma):
    return clip.fl_image(lambda image: gaussian_filter(image, sigma=sigma), apply_to=['mask'])

def merge_special_characters(word_timestamps, original_text):
    """
    Merge special characters with adjacent words based on the original text.
    This ensures that words and punctuation are grouped as they appear in the original text.
    
    Parameters:
        word_timestamps (list): List of dictionaries with 'text', 'offset', and 'duration'.
        original_text (str): The full original text string.
    
    Returns:
        merged_timestamps (list): List of merged dictionaries.
    """
    # Split the original text by spaces to get words with punctuation
    split_words = original_text.split()
    
    merged_timestamps = []
    current_index = 0

    for split_word in split_words:
        merged_text = ""
        merged_start = None
        merged_duration = 0
        
        # Combine entries from word_timestamps to match the split word
        while current_index < len(word_timestamps) and len(merged_text) < len(split_word):
            current_word_info = word_timestamps[current_index]
            word_text = current_word_info['text']
            current_word_info['duration'] = current_word_info['duration'].microseconds

            # Check if current word_text has multiple words (split by spaces)
            word_parts = word_text.split()
            
            # If the current word has multiple parts, handle them
            if len(word_parts) > 1:
                # Keep merging until merged_text length matches the current split_word
                for part in word_parts:
                    if len(merged_text) < len(split_word):
                        if merged_text == "":
                            merged_start = current_word_info['offset']

                        merged_text += part
                        merged_duration += current_word_info['duration']

                        # Add a space if not the last part and split_word expects more characters
                        if len(merged_text) < len(split_word):
                            merged_text += " "
                
                # If we've used up all parts and didn't fully match split_word, adjust index
                if len(merged_text) != len(split_word):
                    current_index -= 1  # Adjust since we only partially consumed this timestamp entry

            else:
                # Start merging the single word entries
                if merged_text == "":
                    merged_start = current_word_info['offset']
                
                merged_text += word_text
                merged_duration += current_word_info['duration']

            # Move to the next word in timestamps
            current_index += 1
        
        # Ensure that we have matched the entire split word
        if merged_text == split_word:
            merged_timestamps.append({
                'text': merged_text,
                'offset': merged_start,
                'duration': merged_duration
            })
        else:
            print(f"ERROR: Unable to match split word '{split_word}' exactly with timestamps.")
            return None

    return merged_timestamps

def get_random_color_combination():
    """Return a random complementary color combination."""
    color_combinations = [
        ('DeepSkyBlue', 'DarkOrange'), 
        ('LimeGreen', 'Purple'), 
        ('Magenta', 'Lime'),
        ('Crimson', 'Aqua'),
        ('Gold', 'RoyalBlue'), 
        ('DodgerBlue', 'Tomato'),
        ('HotPink', 'SpringGreen'), 
        ('OrangeRed', 'Cyan'),
        ('YellowGreen', 'MediumOrchid'), 
        ('Firebrick', 'LightSkyBlue'),
        ('PaleVioletRed', 'Teal'), 
        ('DarkOliveGreen', 'LightCoral'),
        ('SlateBlue', 'Goldenrod'), 
        ('DarkSlateGray', 'LightGoldenrod'),
        ('SeaGreen', 'Orchid'), 
        ('Salmon', 'DarkTurquoise'),
        ('Orange', 'DeepSkyBlue1'), 
        ('ForestGreen', 'LightPink'),
        ('DarkMagenta', 'Chartreuse'), 
        ('MediumSeaGreen', 'Violet')
    ]
    return random.choice(color_combinations)

def zoom_out_effect(clip, zoom_ratio=0.04, target_duration=None):
    """
    Apply a zoom-out effect to a clip over a specified target duration.
    The clip will start at a zoomed-in size and return to normal.
    """
    if target_duration is None:
        target_duration = clip.duration  # Default to full clip duration if not specified

    def effect(get_frame, t):
        # Reverse the zoom factor calculation
        zoom_factor = max(1 - (t / target_duration), 0)  # Clamps to [0, 1] range
        current_zoom_ratio = zoom_ratio * zoom_factor

        img = Image.fromarray(get_frame(t))
        base_size = img.size

        # Calculate the new size based on the current zoom ratio
        new_size = [
            math.ceil(img.size[0] * (1 + current_zoom_ratio)),
            math.ceil(img.size[1] * (1 + current_zoom_ratio))
        ]

        # Ensure even dimensions
        new_size[0] = new_size[0] + (new_size[0] % 2)
        new_size[1] = new_size[1] + (new_size[1] % 2)

        # Resize and crop
        img = img.resize(new_size, Image.LANCZOS)
        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)
        img = img.crop([x, y, new_size[0] - x, new_size[1] - y]).resize(base_size, Image.LANCZOS)

        result = np.array(img)
        img.close()

        return result

    return clip.fl(effect)

def zoom_in_effect(clip, zoom_ratio=0.04, target_duration=None):
    """
    Apply a zoom-in effect to a clip over a specified target duration.
    If target_duration is not provided, the zoom will apply over the entire clip duration.
    """
    if target_duration is None:
        target_duration = clip.duration  # Default to the full clip duration if not specified

    def effect(get_frame, t):
        # Calculate the zoom factor based on the target duration
        zoom_factor = min(t / target_duration, 1)  # Clamp to [0, 1] range
        current_zoom_ratio = zoom_ratio * zoom_factor

        img = Image.fromarray(get_frame(t))
        base_size = img.size

        # Calculate the new size based on the current zoom ratio
        new_size = [
            math.ceil(img.size[0] * (1 + current_zoom_ratio)),
            math.ceil(img.size[1] * (1 + current_zoom_ratio))
        ]

        # Ensure the new dimensions are even numbers
        new_size[0] = new_size[0] + (new_size[0] % 2)
        new_size[1] = new_size[1] + (new_size[1] % 2)

        # Resize and crop the image to achieve the zoom effect
        img = img.resize(new_size, Image.LANCZOS)
        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)
        img = img.crop([x, y, new_size[0] - x, new_size[1] - y]).resize(base_size, Image.LANCZOS)

        result = np.array(img)
        img.close()

        return result

    return clip.fl(effect)

def move_effect(clip, direction, move_ratio=0.1, target_duration=None):
    """
    Apply a movement effect to a clip while zooming in slightly to prevent black bars.
    """
    if target_duration is None:
        target_duration = clip.duration

    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size
        move_factor = min(t / target_duration, 1)  # Normalize time factor
        offset_x, offset_y = 0, 0

        # Compute movement offsets
        if direction == "move_left":
            offset_x = -int(base_size[0] * move_ratio * move_factor)
        elif direction == "move_right":
            offset_x = int(base_size[0] * move_ratio * move_factor)
        elif direction == "move_up":
            offset_y = -int(base_size[1] * move_ratio * move_factor)
        elif direction == "move_down":
            offset_y = int(base_size[1] * move_ratio * move_factor)

        # Zoom into the image to compensate for movement
        zoom_factor = 1 + 2 * move_ratio
        new_size = (int(base_size[0] * zoom_factor), int(base_size[1] * zoom_factor))
        img = img.resize(new_size, Image.LANCZOS)
        
        # Crop the zoomed-in image to maintain original dimensions
        crop_x = (new_size[0] - base_size[0]) // 2
        crop_y = (new_size[1] - base_size[1]) // 2
        img = img.crop((crop_x + offset_x, crop_y + offset_y, crop_x + base_size[0] + offset_x, crop_y + base_size[1] + offset_y))
        
        return np.array(img)

    return clip.fl(effect)

def clip_typewriter(text, text_color, stroke_color, duration_clip, duration_effect, font_size=45, width=500):
    """
    Creates a typewriter effect for the given text.
    - If the text contains one word, it animates each letter.
    - If the text contains multiple words, it animates each word.
    """
    # Set ImageMagick binary path (adjust as per your system)
    change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

    # Determine if the text is a single word or multiple words
    size=(width, 550)
    # Use findObjects to locate each letter or word
    text_clip = TextClip(
        text,
        fontsize=font_size,
        stroke_color=stroke_color,
        stroke_width=2,
        color=text_color,
        font="Trueno_bold",
        method="caption",
        kerning=2,
        size=size
        )
        # If it's a single word, animate each letter
    objects = findObjects(text_clip, preview=False)

    # Select the start time for each letter or word found
    n = len(objects)
    if n > 1:
        times_start = [duration_effect * i / (n - 1) for i in range(n)]
    else:
        times_start = [0]
    clips = []

    for i, obj in enumerate(objects):
        clips.append(obj
            .set_position(obj.screenpos)
            .set_start(times_start[i])
            .set_end(duration_clip)
        )

    # Return the final composite clip
    composite = CompositeVideoClip(clips, size=size).set_position(("center", "center"))
    # composite.write_videofile("out/typewriter.mp4", codec="libx264", audio_codec="aac", fps=18)
    return composite


def smooth_resize(t, duration=0.2, min_scale=0.8, max_scale=1.1):
    """
    Smoothly interpolates scale over time to create a 'jumping' effect.
    
    :param t: Current time in seconds
    :param duration: Duration of the jump effect
    :param min_scale: Minimum scale factor (smaller size)
    :param max_scale: Maximum scale factor (larger size)
    :return: Scale factor for resizing
    """
    if t < duration:
        val = min_scale + (max_scale - min_scale) * (np.sin((t / duration) * np.pi))
        return val
    return 1  # Normal size after animation



def clip_text(text, text_color, stroke_color, duration_clip, font_size=45, width=500):

    change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

    # Determine if the text is a single word or multiple words
    size=(width, 550)
    # Use findObjects to locate each letter or word
    text_clip = TextClip(
        text,
        fontsize=font_size,
        stroke_color=stroke_color,
        stroke_width=2,
        color=text_color,
        font="Trueno_bold",
        method="caption",
        kerning=-1,
        size=size
        ).set_position(("center", "center")).set_duration(duration_clip)
    
    zoom_in_duration = 0.075
    zoomed_in_clip = zoom_in_effect(text_clip, zoom_ratio=0.15, target_duration=zoom_in_duration)

    # Apply zoom out for the remaining duration
    zoom_out_duration = duration_clip - zoom_in_duration
    zoomed_out_clip = zoom_out_effect(zoomed_in_clip, zoom_ratio=0.13, target_duration=zoom_in_duration * 3.5)

    #zoomed_out_clip.write_videofile("out/jump.mp4", codec="libx264", audio_codec="aac", fps=18)

    return text_clip


def parse_srt_file(srt_input):
    """
    Parses the SRT input, which can either be a file path or a JSON array of objects.
    Returns a list of dictionaries with 'text', 'offset', and 'duration' keys.
    """
    # Check if the input is a path to an SRT file
    if isinstance(srt_input, str) and os.path.exists(srt_input):
        # Parse the SRT file using the provided function logic
        srt_subs = pysrt.open(srt_input)
        word_timestamps = []

        for sub in srt_subs:
            start_time = (
                sub.start.hours * 3600 +
                sub.start.minutes * 60 +
                sub.start.seconds +
                sub.start.milliseconds / 1000
            )
            end_time = (
                sub.end.hours * 3600 +
                sub.end.minutes * 60 +
                sub.end.seconds +
                sub.end.milliseconds / 1000
            )
            duration = end_time - start_time

            word_info = {
                'text': sub.text.strip(),
                'offset': start_time,  # Keep offset in seconds for consistency
                'duration': duration   # Duration in seconds
            }
            word_timestamps.append(word_info)

        return word_timestamps

    # Check if the input is a JSON-like object (array of dictionaries)
    elif isinstance(srt_input, list) and all(isinstance(item, dict) for item in srt_input):
        normalized_timestamps = []

        for item in srt_input:
            # Ensure normalization of offset and duration
            if 'offset' in item and 'duration' in item:
                normalized_item = {
                    'text': item.get('text', '').strip(),
                    'offset': item['offset'] / 1_000_000,  # Convert microseconds to seconds
                    'duration': item['duration'].total_seconds()  # Convert timedelta to seconds
                    if isinstance(item['duration'], datetime.timedelta) else item['duration']
                }
                normalized_timestamps.append(normalized_item)
            else:
                raise ValueError("Invalid JSON object. Missing 'offset' or 'duration' keys.")

        return normalized_timestamps

    else:
        raise ValueError("Invalid SRT input. Must be a file path or a JSON array of objects.")

def create_text_clips(srt_input, audio_end, text_color, stroke_color, max_length=57, timing_multiplier=1, font_size=45, width=500, target_text_duration=1):
    """
    Create text clips from timestamps with special effects.
    Ensure each clip has a minimum duration of target_text_duration seconds by combining words if necessary.
    """
    # Parse the SRT input (either file or JSON)
    word_timestamps = parse_srt_file(srt_input)
    if not word_timestamps:
        return None

    text_clips = []
    i = 0
    total_words = len(word_timestamps)
    pbar = tqdm.tqdm(total=total_words)
    while i < total_words:
        # Initialize variables for collecting words
        text_fragment = []
        start_time = word_timestamps[i]['offset'] * timing_multiplier
        word_start_time = start_time
        combined_duration = 0.0

        # Collect words until the combined duration is at least target_text_duration long
        while (i < total_words and combined_duration < target_text_duration) or (i < total_words and word_timestamps[i]["text"].startswith("-")) or (i < total_words and word_timestamps[i]['duration'] * timing_multiplier > target_text_duration and len(text_fragment) == 0):
            word_info = word_timestamps[i]
            word = word_info['text']

            # Check if the word starts with "-" and merge it with the previous word if applicable
            if word.startswith('-') and text_fragment:
                text_fragment[-1] += word.replace("-", "")
            else:
                text_fragment.append(word)

            # Update the duration for the current group of words
            current_word_duration = word_info['duration'] * timing_multiplier
            combined_duration += current_word_duration

            i += 1
            pbar.update(1)

            if (combined_duration >= target_text_duration / 1.5 and word.strip().endswith((",", ".", "!", "?"))):
                break

        # Combine the collected words into a single string
        combined_text = ' '.join(text_fragment)

        # Special case for handling specific abbreviations
        if combined_text.replace("-", "") in ["AITA", "WIBTA", "AITAH", "WIBTAH"]:
            combined_text = combined_text.replace("-", "")

        # Create the TextClip with specified colors and font settings
        change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
        if i == total_words - 1:
            combined_duration = combined_duration + max(0, audio_end - (word_start_time + combined_duration))
            # text_clip = clip_typewriter(combined_text, text_color, stroke_color, combined_duration, combined_duration / 3, font_size=font_size, width=width)
            text_clip = clip_text(combined_text, text_color, stroke_color, combined_duration, font_size=font_size, width=width)
        else:
            text_clip = clip_text(combined_text, text_color, stroke_color, combined_duration, font_size=font_size, width=width)

        # Apply text clip settings
        text_clip = text_clip.set_start(word_start_time)
        text_clip = text_clip.set_position(('center', 'center'))
        text_clip = text_clip.set_duration(combined_duration)
        # if text_clip.start + text_clip.duration < short_duration:
        #     text_clips_short.append(text_clip)

        text_clips.append(text_clip)
        if text_clip.start + text_clip.duration > max_length:
            return text_clips

    return text_clips

def get_audio_duration(audio_file):
    if audio_file == None:
        return -1
    """ Get the duration of the audio file in seconds using MoviePy. """
    with AudioFileClip(audio_file) as audio_clip:
        return audio_clip.duration

def loop_video_clip(video_clip, min_duration=30):
    """ Loop a video clip until it exceeds the specified duration. """
    video_duration = video_clip.duration
    if video_duration >= min_duration:
        return video_clip
    # Repeat the video clip until it exceeds min_duration
    loop_count = int(min_duration // video_duration) + 1
    return concatenate_videoclips([video_clip] * loop_count).subclip(0, min_duration)

def calculate_black_pixel_percentage(frame, rgb_threshold=(0, 0, 0)):
    """
    Calculate the percentage of purely black pixels in a given frame.

    Parameters:
        frame (numpy.ndarray): The frame image (RGB format).
        rgb_threshold (tuple): The RGB intensity threshold to consider a pixel as black.

    Returns:
        float: The percentage of purely black pixels in the frame.
    """
    # Check for purely black pixels by comparing each RGB channel to the threshold
    black_pixels = np.all(frame < rgb_threshold, axis=2)
    total_pixels = frame.shape[0] * frame.shape[1]
    black_pixel_count = np.sum(black_pixels)

    return (black_pixel_count / total_pixels) * 100

def add_overlay_video(base_video, overlay_video_path, start_time, output_video, final_audio=None):
    """
    Overlay another video (with audio) on top of the base video at a specific timestamp.
    
    Parameters:
    - base_video: The original composite video (VideoFileClip or CompositeVideoClip).
    - overlay_video_path: Path to the overlay video file.
    - start_time: Timestamp (in seconds) where the overlay video should start.
    - output_video: Path where the final video will be saved.
    - final_audio: Optional audio clip to set as the final audio.
    """
    
    # Load the overlay video
    overlay_clip = VideoFileClip(overlay_video_path)
    overlay_clip = overlay_clip.resize(width=2 * base_video.w // 3)
    
    # Adjust the start time of the overlay video
    overlay_clip = overlay_clip.set_start(start_time)
    
    # Set position if needed (e.g., center it)
    overlay_clip = overlay_clip.set_position(("center", "center"))
    
    # Combine the base video and overlay video
    final_composite = CompositeVideoClip([base_video, overlay_clip])
    
    # Optionally set audio (if provided)
    if final_audio:
        final_composite = final_composite.set_audio(final_audio)
    
    # Write the final video file
    final_composite.write_videofile(output_video, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True, fps=FPS, threads = 24)
    logging.info(f"Final video saved at {output_video}")

def create_silence(duration=3):
    return AudioClip(lambda t: 0, duration=duration, fps=44100)


def calculate_new_resolution(current_width, current_height, target_aspect_ratio):
    """
    Calculate the new resolution to fit a target aspect ratio while minimizing cropping.

    Args:
        current_width (int): Original video width.
        current_height (int): Original video height.
        target_aspect_ratio (tuple): Target aspect ratio as (width, height).

    Returns:
        tuple: (new_width, new_height)
    """
    target_ar = target_aspect_ratio[0] / target_aspect_ratio[1]
    current_ar = current_width / current_height

    if current_ar > target_ar:
        # Original is wider → crop width
        new_width = round(current_height * target_ar)
        return new_width, current_height
    else:
        # Original is taller → crop height
        new_height = round(current_width / target_ar)
        return current_width, new_height

def compile_and_resize_videos(total_duration, target_folder="data/temp", aspect_ratio=(9, 16)):
    """
    Compile and resize video clips or images into a final video with the specified duration.
    
    If images are found, they are converted to videos with random camera effects.
    
    Args:
        total_duration (float): Total duration of the final video in seconds.
        target_folder (str): Folder containing video clips or images.
        aspect_ratio (tuple): Aspect ratio for resizing (width, height).
        resolution (tuple): Desired resolution for the final video (width, height).
    
    Returns:
        list: List of processed video clips.
    """
    if not os.path.exists(target_folder):
        logging.error(f"Folder '{target_folder}' does not exist.")
        return None

    video_files = [
        os.path.join(target_folder, f)
        for f in os.listdir(target_folder)
        if f.endswith((".mp4", ".mkv", ".mov"))
    ]
    image_files = [
        os.path.join(target_folder, f)
        for f in os.listdir(target_folder)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    
    if not video_files and not image_files:
        logging.error("No valid video or image files found in the target folder.")
        return None
    
    # Sort files by the first number in their name
    def extract_number(filename):
        match = re.search(r"(\d+)", os.path.basename(filename))
        return int(match.group(1)) if match else float("inf")

    video_files.sort(key=extract_number)
    image_files.sort(key=extract_number)
    
    compiled_clips = []
    remaining_duration = total_duration
    total_media_count = len(video_files) + len(image_files)

    
    for i, media_file in enumerate(video_files + image_files):
        resolution = None
        try:
            if media_file in video_files:
                clip = VideoFileClip(media_file)
            else:
                img = ImageClip(media_file)
                resolution = img.size
                del img
                clip = create_image_video_clip(media_file, resolution, duration=total_duration/total_media_count)

            
            # (w, h) = clip.size
            
            # if h != resolution[1] and w != resolution[0]:
            #     clip = clip.resize(height=resolution[1])

            #     new_width = round(resolution[1] * (aspect_ratio[0] / aspect_ratio[1]))

            #     # Resize the width to match the correct aspect ratio
            #     clip = clip.resize(width=new_width)  

            #     # Ensure proper cropping in case there’s extra space
            #     clip = clip.crop(x_center=new_width / 2, y_center=h / 2, width=new_width, height=resolution[1])

            # clip.write_videofile(r"out/temp_video.mp4", codec='libx264')

            
            if i == 0:
                # clip = clip.fx(vfx.speedx, factor=2) $ change to speed up first clip
                first_video_duration = clip.duration
                compiled_clips.append(clip.set_duration(first_video_duration))
                remaining_duration -= first_video_duration
            else:
                segment_duration = remaining_duration / (total_media_count - len(compiled_clips))
                clip = clip.fx(vfx.speedx, clip.duration / segment_duration)
                compiled_clips.append(clip.set_duration(round(segment_duration, 1)))
                
        except Exception as e:
            logging.error(f"Error processing file '{media_file}': {e}")
    
    if len(compiled_clips) < 6:
        logging.error("Not enough valid media files to compile the required duration.")
        return None
    
    logging.info(f"Compiled clips from: {target_folder}")
    return compiled_clips

def create_image_video_clip(image_path, resolution, duration=5):
    """
    Create a video clip from an image with proper cropping and camera movement.
    
    Args:
        image_path (str): Path to the image file.
        resolution (tuple): Target resolution (width, height).
        duration (int): Duration of the image clip in seconds.
    
    Returns:
        ImageSequenceClip: Video clip with animation applied.
    """
    image_clip = ImageClip(image_path, duration=duration).set_fps(FPS)

    # # Original image dimensions
    # img_width, img_height = image_clip.size

    # target_width, target_height = resolution  # e.g., (576, 1024) for vertical video


    # image_clip = image_clip.fx(resize, newsize=(target_width, img_height))

    # if img_width / img_height > target_width / target_height:
    #     new_width = int((img_height / target_height) * target_width)
    #     image_clip = image_clip.crop(
    #         x_center=img_width // 2, width=new_width, height=img_height
    #     )
    # else:
    #     image_clip = image_clip.resize(width=target_width, height=target_height)
    #     image_clip = image_clip.crop(
    #         y_center=img_height // 2, width=target_width, height=target_height
    #     )


    # Apply movement effects
    effect = random.choice(["zoom_in", "zoom_out", "move_left", "move_right", "move_up", "move_down"])

    zoom_ratio=0.4

    movement_clip = None
    if effect == "zoom_in":
        movement_clip = zoom_in_effect(image_clip, zoom_ratio=zoom_ratio, target_duration=duration)
    elif effect == "zoom_out":
        movement_clip = zoom_out_effect(image_clip, zoom_ratio=zoom_ratio, target_duration=duration)
    else:
        movement_clip = move_effect(image_clip, effect, move_ratio=zoom_ratio/2, target_duration=duration)
    return movement_clip

def combine_audio_and_video(
    full_audio_path, video_clips_path, images_path, full_timings, full_text, output_path, output_video="full_video.mp4",
    output_video_short="short_video.mp4", resolution=(1820, 1024)
):
    try:
        # Define full and short video output paths
        output_video = os.path.join(output_path, output_video)
        output_video_short = os.path.join(output_path, output_video_short)

        # Load audio clips and set up final audio
        combined_audio = AudioFileClip(full_audio_path)
        ding_audio = AudioFileClip("data/sound_effects/ding.mp3").volumex(0.6).set_duration(1.5)
        background_audio = AudioFileClip("data/sound_effects/background_audio.mp3").volumex(0.1)
        total_duration = combined_audio.duration

        background_audio_looped = concatenate_audioclips([
            background_audio] * int(total_duration // background_audio.duration + 1))
        background_audio_looped = background_audio_looped.subclip(0, total_duration)

        final_audio = concatenate_audioclips([ding_audio, background_audio_looped])
        final_audio = CompositeAudioClip([combined_audio, final_audio])

        total_duration = round(final_audio.duration, 2)
        short_duration = total_duration + 8
        #short_duration = min(total_duration, 68)
        short_audio_duration = short_duration - 8

        if images_path != None and os.path.exists(images_path):
            full_clips = compile_and_resize_videos(total_duration, images_path)
            short_clips = compile_and_resize_videos(total_duration, images_path)

        else:
            full_clips = compile_and_resize_videos(total_duration, video_clips_path)
            short_clips = compile_and_resize_videos(total_duration, video_clips_path)


        full_video = concatenate_videoclips(full_clips).set_duration(total_duration)
        short_video = concatenate_videoclips(short_clips).set_duration(short_duration)

        target_aspect_ratio = (16, 9)

        video_width, video_height = full_video.size  

        target_height = video_height
        target_width = round(target_height * target_aspect_ratio[0] / target_aspect_ratio[1])
        target_width = max(target_width, video_width)

        full_video = full_video.fx(resize, newsize=(target_width, video_height))
        
        target_aspect_ratio = (9, 16)

        video_width, video_height = short_video.size  

        target_height = video_height
        target_width = round(target_height * target_aspect_ratio[0] / target_aspect_ratio[1])
        target_width = min(target_width, video_width)
        short_video = short_video.subclip(0, short_duration)

        x_center = short_video.w // 2
        x1 = max(0, x_center - target_width // 2)
        x2 = min(video_width, x_center + target_width // 2)

        
        if target_width < video_width:
            cropped_video = short_video.crop(x1=x1, x2=x2, y1=0, y2=target_height)
        else:
            cropped_video = short_video

        short_audio = final_audio.subclip(0, short_audio_duration)


        text_color = "White"
        stroke_color = "Blue"
        logging.info("Creating text clips")

        #full_text_clips = create_text_clips(srt_input=full_timings, audio_end=total_duration, text_color=text_color, stroke_color=stroke_color, max_length=total_duration, font_size=45, width=1000, target_text_duration=1.5)
        short_text_clips = create_text_clips(srt_input=full_timings, audio_end=short_audio_duration, text_color=text_color, stroke_color=stroke_color, max_length=short_audio_duration, font_size=35, target_text_duration=0.4)
       
        logging.info("Text clips completed")


        # full_text_overlay = CompositeVideoClip(full_text_clips).set_position(("center", "bottom"))        
        # full_composite = CompositeVideoClip([full_video, full_text_overlay], use_bgclip=True).set_duration(total_duration)
        # full_composite = full_composite.set_audio(final_audio)
        # full_composite.write_videofile(output_video, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', threads=24, fps=60, remove_temp=True)
        # logging.info(f"Full video saved at {output_video}")

        short_text_overlay = CompositeVideoClip(short_text_clips).set_position(("center", "center"))
        short_composite = CompositeVideoClip([cropped_video, short_text_overlay], use_bgclip=True)
        short_composite = short_composite.subclip(0, short_duration)
        short_composite = short_composite.set_audio(short_audio)

        add_overlay_video(short_composite, r"data\like_part_2.mp4", short_audio.duration, output_video_short)

        #short_composite.write_videofile(output_video_short, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True, fps=30, threads = 24)
        logging.info(f"Short video saved at {output_video_short}")

        return True

    except Exception as e:
        logging.error(f"Error combining audio and video: {traceback.format_exc()}")
        return False

    
def create_combined_video_for_post(post, full, output_folder="out/", video_clips_path="", images_path=None, gen_id=uuid.uuid4()):
    """Create a combined video for the post using the video generator."""
    # total_audio_duration = get_audio_duration(full[0])
    # # Select random videos to match the duration of the combined audio
    # video_clips = select_and_trim_videos(duration_needed=total_audio_duration)
    # if video_clips == None:
    #     return None

    output_video_path = os.path.join(output_folder, f"{gen_id}")
    os.makedirs(output_video_path, exist_ok=True)

    if combine_audio_and_video(
        # title_audio_file=title_tts_audio_file,
        # content_audio_file=content_tts_audio_file,
        full_audio_path = full[0],
        video_clips_path=video_clips_path,
        images_path=images_path,
        # title_timings=title[1],   # Word timings for title TTS
        # title_text=title[2],
        # content_timings=content[1], # Word timings for content TTS
        # content_text=content[2],
        full_timings = full[1],
        full_text = full[2],
        output_path=output_video_path,
    ):
        return True
    else:
        logging.error(f"Could not create video for post: {post['title']}")       