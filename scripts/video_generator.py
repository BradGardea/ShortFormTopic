from moviepy.editor import *
from moviepy.config import change_settings
from scipy.ndimage import gaussian_filter
import random
import os
import logging
from datetime import timedelta
import numpy as np


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

import random
from moviepy.editor import TextClip

def create_text_clips(word_timestamps, original_text, text_color, stroke_color):
    """
    Create text clips from timestamps with special effects.
    Ensure each clip has a minimum duration of 0.4 seconds by combining words if necessary.
    """
    merged_word_timestamps = merge_special_characters(word_timestamps, original_text)
    if merged_word_timestamps is None:
        return None

    text_clips = []
    i = 0
    total_words = len(merged_word_timestamps)

    while i < total_words:
        # Initialize variables for collecting words
        text_fragment = []
        start_time = merged_word_timestamps[i]['offset'] / 1_000_000  # Convert to seconds
        word_start_time = start_time
        combined_duration = 0.0
        
        # Collect words until the combined duration is at least 0.4 seconds
        while i < total_words and combined_duration < 0.4:
            word_info = merged_word_timestamps[i]
            word = word_info['text']
            text_fragment.append(word)

            # Update the duration for the current group of words
            current_word_duration = word_info['duration'] / 1_000_000
            combined_duration += current_word_duration
            
            i += 1

        # Combine the collected words into a single string
        combined_text = ' '.join(text_fragment)

        # Create the TextClip with specified colors and font settings
        change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
        text_clip = TextClip(
            combined_text,
            fontsize=120,
            stroke_color=stroke_color,
            stroke_width=3,
            color=text_color,
            font="Trueno_bold",
            method="label",
            kerning=-1,
            # size=(1080, 200)  # Adjusted size for better visibility
        )

        # Apply the zoom-in effect (starts at 50% size and grows to full size)
        text_clip = text_clip.set_start(word_start_time)
        text_clip = text_clip.set_position(('center', 'center'))
        text_clip = text_clip.set_duration(combined_duration)

        # Add the zoom-in jump effect (starts at 50% size and grows to full size)
        zoomed_clip = text_clip.resize(lambda t: 0.1 + 0.5 * (t / text_clip.duration)).crossfadein(0.4)

        # Append the created clip to the list
        text_clips.append(zoomed_clip)

    return text_clips


def get_audio_duration(audio_file):
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


def select_and_trim_videos(duration_needed, video_folder="data/satisfying_videos"):
    """ 
    Select random videos from 'loops' and 'regular' folders, checking for black pixel content, 
    and trim them to fit the needed duration.
    """
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

        while video_clip is None:
            if curr_checks >= max_checks:
                return None
            else:
                try:
                    video_file = random.choice(video_files)
                    video_clip = VideoFileClip(video_file)
                    video_duration = video_clip.duration
                    
                    # Check for too many black pixels
                    if video_has_too_many_black_pixels(video_clip):
                        logging.info(f"Video '{video_file}' rejected due to high black pixel content.")
                        # video_files.remove(video_file)
                        video_clip = None
                        continue
                    logging.info(f"Video '{video_file}' selected.")


                except Exception as e:
                    logging.error(f"Error processing video '{video_file}': {e}")
                    #video_files.remove(video_file)
                    video_clip = None

                curr_checks += 1

        # Check if the video is from the 'loops' folder and loop it if it's less than 30 seconds
        if 'loops' in video_file and video_duration < 30:
            video_clip = loop_video_clip(video_clip, 40)
            video_duration = video_clip.duration

        if total_duration + video_duration <= duration_needed:
            selected_videos.append(video_clip)
            total_duration += video_duration
        else:
            # If adding this video exceeds the duration needed, trim it
            trimmed_video = video_clip.subclip(0, duration_needed - total_duration)
            selected_videos.append(trimmed_video)
            total_duration += trimmed_video.duration
            trimmed_video.close()
            break  # Stop once we've reached the required duration

    return selected_videos


def combine_audio_and_video(title_audio_file, content_audio_file, video_clips, title_timings, title_text, content_timings, content_text, output_video="final_output.mp4"):
    """Combine audio files and video clips with text overlays into a final video."""
    # Load the audio files
    title_audio_clip = AudioFileClip(title_audio_file)
    content_audio_clip = AudioFileClip(content_audio_file)
    
    # Concatenate the audio clips (title followed by content)
    combined_audio = concatenate_audioclips([title_audio_clip, content_audio_clip])
    
    # Calculate total audio duration
    total_duration = combined_audio.duration

    background_audio_clip = AudioFileClip("data/background_audio.mp3")
    background_audio_looped = concatenate_audioclips([background_audio_clip] * int(total_duration // background_audio_clip.duration + 1))
    background_audio_looped = background_audio_looped.subclip(0, total_duration).volumex(0.2)  # Lower the volume of background music
    final_audio = CompositeAudioClip([combined_audio, background_audio_looped])

    
    text_color, stroke_color = get_random_color_combination()

    # Generate text overlays
    title_text_clips = create_text_clips(title_timings, title_text, text_color, stroke_color)
    if title_text == None:
        return None
    content_text_clips = create_text_clips(content_timings[: 100], content_text[:100], text_color, stroke_color)
    if content_text_clips == None:
        return None

    
    new_clips = []
    for clip in video_clips:
        (w, h) = clip.size
        clip = clip.resize(height=1920)
        logging.info(f"Resized clip size: {clip.w}x{clip.h}")
        adjusted_center = ((1920 / h) * w) / 2
        new_clip = clip.crop(x_center=adjusted_center, y_center=960, width=1080, height=1920)
        new_clips.append(new_clip)
        logging.info(f"Cropped clip size: {new_clip.w}x{new_clip.h}")


    final_video = concatenate_videoclips(new_clips).set_duration(total_duration)

    title_clip = concatenate_videoclips(title_text_clips, method="compose")
    content_clip = concatenate_videoclips(content_text_clips, method="compose")
    if title_text_clips:
        last_title_clip_end = title_text_clips[-1].end
    else:
        last_title_clip_end = 0

    title_audio_end = title_audio_clip.duration
    gap_duration = max(0, title_audio_end - last_title_clip_end) * 4
    content_clip.set_start(content_clip.start + gap_duration)
    text_clip = concatenate_videoclips([title_clip, content_clip], method="compose").set_position(('center', 'center'))

    composite_video = CompositeVideoClip([final_video, text_clip]).set_position(('center', 'center'))

    composite_video = composite_video.set_audio(final_audio)
    composite_video = composite_video.subclip(0, 10)
    composite_video.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=18)
    logging.info(f"Final video saved at {output_video}")

def create_combined_video_for_post(post, title, content, output_folder="out/"):
    """Create a combined video for the post using the video generator."""
    # Calculate total audio duration
    title_tts_audio_file = title[0]
    content_tts_audio_file = content[0]
    title_duration = get_audio_duration(title_tts_audio_file)
    content_duration = get_audio_duration(content_tts_audio_file)
    total_audio_duration = title_duration + content_duration + 5  # Adding buffer time

    # Select random videos to match the duration of the combined audio
    video_clips = select_and_trim_videos(duration_needed=total_audio_duration)
    if video_clips == None:
        return

    # Generate the final video with text overlays
    output_video_path = os.path.join(output_folder, f"{post['id']}_final_video.mp4")
    if combine_audio_and_video(
        title_audio_file=title_tts_audio_file,
        content_audio_file=content_tts_audio_file,
        video_clips=video_clips,
        title_timings=title[1],   # Word timings for title TTS
        title_text=title[2],
        content_timings=content[1], # Word timings for content TTS
        content_text=content[2],
        output_video=output_video_path
    ) == None:
        logging.error(f"Could not create video for post: {post['title']}")
        return None
    
    logging.info(f"Video created for post: {post['title']} at {output_video_path}")
    return True