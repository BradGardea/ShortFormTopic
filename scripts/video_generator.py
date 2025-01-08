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

def zoom_out_effect(clip, zoom_max_ratio=0.2, zoom_out_factor=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size
 
        # Reverse the zoom effect by starting zoomed in and zooming out
        scale_factor = zoom_max_ratio - (zoom_out_factor * t)
        scale_factor = max(scale_factor, 0)  # Ensure scale factor doesn't go negative
 
        new_size = [
            math.ceil(base_size[0] * (1 + scale_factor)),
            math.ceil(base_size[1] * (1 + scale_factor))
        ]
 
        # The new dimensions must be even.
        new_size[0] = new_size[0] - (new_size[0] % 2)
        new_size[1] = new_size[1] - (new_size[1] % 2)
 
        img = img.resize(new_size, Image.LANCZOS)
 
        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)
 
        img = img.crop([
            x, y, new_size[0] - x, new_size[1] - y
        ])
 
        # Resize back to base size
        img = img.resize(base_size, Image.LANCZOS)
 
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

def clip_typewriter(text, text_color, stroke_color, duration_clip, duration_effect):
    """
    Creates a typewriter effect for the given text.
    - If the text contains one word, it animates each letter.
    - If the text contains multiple words, it animates each word.
    """
    # Set ImageMagick binary path (adjust as per your system)
    change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

    # Determine if the text is a single word or multiple words
    size=(1080, 500)
    # Use findObjects to locate each letter or word
    text_clip = TextClip(
        text,
        fontsize=150,
        stroke_color=stroke_color,
        stroke_width=5,
        color=text_color,
        font="Trueno_bold",
        method="caption",
        kerning=-2,
        size=size  # Adjusted size for better visibility
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

def parse_srt_file(srt_path):
    """
    Parses the SRT file and extracts word-level timestamps.
    Returns a list of dictionaries with 'text', 'offset', and 'duration' keys.
    """
    srt_subs = pysrt.open(srt_path)
    word_timestamps = []

    for sub in srt_subs:
        start_time = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000)
        end_time = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000)
        duration = end_time - start_time

        word_info = {
            'text': sub.text.strip(),
            'offset': start_time,  # Keep offset in seconds for consistency
            'duration': duration   # Duration in seconds
        }
        word_timestamps.append(word_info)

    return word_timestamps

def create_text_clips(srt_path, audio_end, text_color, stroke_color, max_length = 57, timing_multiplier=1):
    """
    Create text clips from timestamps with special effects.
    Ensure each clip has a minimum duration of 0.2 seconds by combining words if necessary.
    """
    # Parse the SRT file to get word timestamps
    word_timestamps = parse_srt_file(srt_path)
    if not word_timestamps:
        return None

    text_clips = []
    i = 0
    total_words = len(word_timestamps)
    while i < total_words:
        # Initialize variables for collecting words
        text_fragment = []
        start_time = word_timestamps[i]['offset'] * timing_multiplier
        word_start_time = start_time
        combined_duration = 0.0
        
        # Collect words until the combined duration is at least 0.5 seconds
        while (i < total_words and combined_duration < 0.12) or (i < total_words and word_timestamps[i]["text"].startswith("-")):
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

        # Combine the collected words into a single string
        combined_text = ' '.join(text_fragment)

        # Special case for handling specific abbreviations
        if combined_text.replace("-", "") in ["AITA", "WIBTA", "AITAH", "WIBTAH"]:
            combined_text = combined_text.replace("-", "")

        # Create the TextClip with specified colors and font settings
        change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
        if i == total_words - 1:
            combined_duration = combined_duration + max(0, audio_end - (word_start_time + combined_duration))
            text_clip = clip_typewriter(combined_text, text_color, stroke_color, combined_duration, combined_duration / 2.5)
        else:
            text_clip = clip_typewriter(combined_text, text_color, stroke_color, combined_duration, combined_duration / 2.5)
        
        # Apply text clip settings
        text_clip = text_clip.set_start(word_start_time)
        text_clip = text_clip.set_position(('center', 'center'))
        text_clip = text_clip.set_duration(combined_duration)
        # text_clip.write_videofile("out/typewriter.mp4", codec="libx264", audio_codec="aac", fps=18)
        # print(f"Added text: {combined_text} from {text_clip.start} to {text_clip.start +text_clip.duration}\n")
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
    overlay_clip = overlay_clip.resize(height=800)
    
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
    final_composite = final_composite.subclip(0, 59)
    final_composite.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=18)
    print(f"Final video saved at {output_video}")

def create_silence(duration=3):
    return AudioClip(lambda t: 0, duration=duration, fps=44100)


def compile_and_resize_videos(total_duration, target_folder="data/temp", aspect_ratio=(9, 16), resolution=(512, 512)):
    """
    Compile and resize video clips to match the desired duration, splitting the total duration into equal parts.

    Args:
        total_duration (float): Total duration of the final video in seconds.
        target_folder (str): Folder containing video clips.
        aspect_ratio (tuple): Aspect ratio for resizing (width, height).
        resolution (tuple): Desired resolution for the final video (width, height).

    Returns:
        list: List of resized and duration-adjusted video clips.
    """
    if not os.path.exists(target_folder):
        logging.error(f"Folder '{target_folder}' does not exist.")
        return None

    video_files = [
        os.path.join(target_folder, f)
        for f in os.listdir(target_folder)
        if f.endswith((".mp4", ".mkv", ".mov"))
    ]

    if not video_files:
        logging.error("No valid video files found in the target folder.")
        return None

    # Sort video files by the first number in their name
    def extract_number(filename):
        match = re.search(r"(\d+)", os.path.basename(filename))
        return int(match.group(1)) if match else float("inf")

    video_files.sort(key=extract_number)

    # Calculate the duration each clip should have
    segment_duration = total_duration / 6
    compiled_clips = []

    for video_file in video_files:
        try:
            video_clip = VideoFileClip(video_file)
            video_clip = video_clip.resize(height=resolution[1])
            (w, h) = video_clip.size

            if h != resolution[1] or w / h != aspect_ratio[0] / aspect_ratio[1]:
                new_width = (resolution[1] * aspect_ratio[0]) / aspect_ratio[1]
                video_clip = video_clip.crop(
                    x_center=w / 2, y_center=h / 2, width=new_width, height=resolution[1]
                )

            video_clip = video_clip.fx(vfx.speedx, video_clip.duration / segment_duration)
            compiled_clips.append(video_clip.set_duration(segment_duration))

            if len(compiled_clips) == 6:
                break

        except Exception as e:
            logging.error(f"Error processing video '{video_file}': {e}")

    if len(compiled_clips) < 6:
        logging.error("Not enough valid videos to compile the required duration.")
        return None

    return compiled_clips

def compile_and_resize_videos(total_duration, target_folder="data/temp", aspect_ratio=(9, 16), resolution=(512, 512)):
    if not os.path.exists(target_folder):
        logging.error(f"Folder '{target_folder}' does not exist.")
        return None

    video_files = [
        os.path.join(target_folder, f)
        for f in os.listdir(target_folder)
        if f.endswith((".mp4", ".mkv", ".mov"))
    ]

    if not video_files:
        logging.error("No valid video files found in the target folder.")
        return None

    def extract_number(filename):
        match = re.search(r"(\d+)", os.path.basename(filename))
        return int(match.group(1)) if match else float("inf")

    video_files.sort(key=extract_number)

    segment_duration = total_duration / 6
    compiled_clips = []

    for video_file in video_files:
        try:
            video_clip = VideoFileClip(video_file)
            video_clip = video_clip.resize(height=resolution[1])
            (w, h) = video_clip.size

            if h != resolution[1] or w / h != aspect_ratio[0] / aspect_ratio[1]:
                new_width = (resolution[1] * aspect_ratio[0]) / aspect_ratio[1]
                video_clip = video_clip.crop(
                    x_center=w / 2, y_center=h / 2, width=new_width, height=resolution[1]
                )

            video_clip = video_clip.fx(vfx.speedx, video_clip.duration / segment_duration)
            compiled_clips.append(video_clip.set_duration(segment_duration))

            if len(compiled_clips) == 6:
                break

        except Exception as e:
            logging.error(f"Error processing video '{video_file}': {e}")

    if len(compiled_clips) < 6:
        logging.error("Not enough valid videos to compile the required duration.")
        return None

    return compiled_clips


def combine_audio_and_video(
    full_audio_path, video_clips_path, full_timings, full_text, output_path, output_video_short="short_video.mp4",
    output_video_long="long_video.mp4", resolution=(868, 480)
):
    try:
        output_video_long = os.path.join(output_path, output_video_long)
        output_video_short = os.path.join(output_path, output_video_short)
    
        combined_audio = AudioFileClip(full_audio_path)
        ding_audio = AudioFileClip("data/sound_effects/ding.mp3").volumex(0.8)
        background_audio = AudioFileClip("data/sound_effects/background_audio.mp3").volumex(0.2)
        total_duration = combined_audio.duration

        background_audio_looped = concatenate_audioclips([
            background_audio] * int(total_duration // background_audio.duration + 1))
        background_audio_looped = background_audio_looped.subclip(0, total_duration)

        final_audio = concatenate_audioclips([ding_audio, background_audio_looped])
        final_audio = CompositeAudioClip([combined_audio, final_audio])

        short_clips = compile_and_resize_videos(60, video_clips_path, (9, 16), resolution)
        long_clips = compile_and_resize_videos(total_duration, video_clips_path, (1, 1), resolution)

        if not short_clips or not long_clips:
            return False

        text_color, stroke_color = get_random_color_combination()
        short_text_clips = create_text_clips(full_timings, 60, text_color, stroke_color, 60)
        long_text_clips = create_text_clips(full_timings, total_duration, text_color, stroke_color, total_duration)

        short_video = concatenate_videoclips(short_clips).set_duration(60)
        short_text_overlay = CompositeVideoClip(short_text_clips).set_position(("center", "center"))
        short_composite = CompositeVideoClip([short_video, short_text_overlay], use_bgclip=True)
        short_composite = short_composite.set_audio(final_audio.subclip(0, 60))
        short_composite.write_videofile(output_video_short, codec="libx264", audio_codec="aac", fps=18)
        logging.info(f"Short video saved at {output_video_short}")

        long_video = concatenate_videoclips(long_clips).set_duration(total_duration)
        long_text_overlay = CompositeVideoClip(long_text_clips).set_position(("center", "center"))
        long_composite = CompositeVideoClip([long_video, long_text_overlay], use_bgclip=True)
        long_composite = long_composite.set_audio(final_audio)
        long_composite.write_videofile(output_video_long, codec="libx264", audio_codec="aac", fps=18)
        logging.info(f"Long video saved at {output_video_long}")

        return True

    except Exception as e:
        logging.error(f"Error combining audio and video: {e}")
        return False

def create_combined_video_for_post(post, full, output_folder="out/", video_clips_path=""):
    """Create a combined video for the post using the video generator."""
    # total_audio_duration = get_audio_duration(full[0])
    # # Select random videos to match the duration of the combined audio
    # video_clips = select_and_trim_videos(duration_needed=total_audio_duration)
    # if video_clips == None:
    #     return None

    output_video_path = os.path.join(output_folder, f"{post['id']}")
    os.makedirs(output_video_path, exist_ok=True)

    if combine_audio_and_video(
        # title_audio_file=title_tts_audio_file,
        # content_audio_file=content_tts_audio_file,
        full_audio_path = full[0],
        video_clips_path=video_clips_path,
        # title_timings=title[1],   # Word timings for title TTS
        # title_text=title[2],
        # content_timings=content[1], # Word timings for content TTS
        # content_text=content[2],
        full_timings = full[1],
        full_text = full[2],
        output_path=output_video_path,
    ) == None:
        logging.error(f"Could not create video for post: {post['title']}")
        return None