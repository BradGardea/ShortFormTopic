from moviepy.editor import *
from moviepy.config import change_settings
from scipy.ndimage import gaussian_filter
import random
import os
import logging
from datetime import timedelta


def blur(clip, sigma):
    return clip.fl_image(lambda image: gaussian_filter(image, sigma=sigma), apply_to=['mask'])

def create_text_overlay_clips(word_timestamps, duration):
    """
    Generate text overlay clips based on word timings.
    Each word appears at its specified offset for its duration.
    """
    text_clips = []
    for word_info in word_timestamps:
        text = word_info['text']
        start_time = word_info['offset'] / 1000  # Convert to seconds
        word_duration = word_info['duration'].microseconds / 100 # Convert timedelta to seconds
        change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

        # Create a text clip that appears at the center of the screen
        text_clip = TextClip(text, fontsize=80, color='white', filename="C:\\Windows\\Fonts\\impact.ttf", method="caption", align="center")
        text_clip = text_clip.set_start(start_time)
        text_clip = text_clip.set_position(('center', 'center'))
        text_clip = text_clip.set_duration(word_duration)
                     
        shadow = text_clip.copy()
        shadow = shadow.set_position(('center', 'center'))
        shadow.color = 'black'
        shadow = blur(shadow, sigma=5)
        # composite = CompositeVideoClip([shadow.set_position(('center', 'center')), text_clip.set_position(('center', 'center'))]).set_position(('center', 'center'))
        composite = CompositeVideoClip([shadow, text_clip]).set_position(('center', 'center'))

        text_clips.append(composite)
    
    return text_clips

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
        merged_duration = timedelta(0)
        
        # Combine entries from word_timestamps to match the split word
        while current_index < len(word_timestamps) and len(merged_text) < len(split_word):
            current_word_info = word_timestamps[current_index]
            word_text = current_word_info['text']

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

def create_text_clips(word_timestamps, original_text):
    """Create text clips from timestamps."""
    merged_word_timestamps = merge_special_characters(word_timestamps, original_text)
    if merged_word_timestamps == None:
        return None
    text_clips = []
    
    for word_info in merged_word_timestamps:
        text = word_info['text']
        start_time = word_info['offset'] / 1000  # Convert to seconds
        word_duration = word_info['duration'].microseconds / 90
        print(f"Adding text: {text} from {start_time} to {start_time + word_duration}")
        change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
        # Create a text clip that appears at the center of the screen
        text_clip = TextClip(text, fontsize=150, color='white', filename=r"C:\Users\brad8\AppData\Local\Microsoft\Windows\Fonts\TooneyNoodleNF.ttf", method="caption", align="center")
        text_clip = text_clip.set_start(start_time)
        text_clip = text_clip.set_position(('center', 'center'))
        text_clip = text_clip.set_duration(word_duration)
                     
        # # Create a shadow effect
        # shadow = TextClip(text, fontsize=100, color='black', font="C:\\Windows\\Fonts\\impact.ttf", method="caption", align="center")
        # shadow = text_clip.set_start(start_time)
        # shadow = text_clip.set_position(('center', 'center'))
        # shadow = text_clip.set_duration(word_duration)
        # shadow = blur(shadow, sigma=5)
        
        # # Create composite clip with text and shadow
        # composite = CompositeVideoClip([shadow, text_clip]).set_position(('center', 'center'))
        # composite.start = start_time
        # composite.duration = word_duration
        # text_clips.append(composite)

        text_clips.append(text_clip)
    
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

def select_and_trim_videos(duration_needed, video_folder="data/satisfying_videos"):
    """ Select random videos from 'loops' and 'regular' folders and trim them to fit the needed duration. """
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
        while video_clip == None:
            if curr_checks >= max_checks:
                return None
            else:
                try:
                    video_file = random.choice(video_files)
                    video_clip = VideoFileClip(video_file)
                    video_duration = video_clip.duration
                    # w, h = video_clip.size
                    # if w + 75 < 1080 or h + 140 < 1920:
                    #     #TODO: remove small videos
                    #     logging.info(f"Video too small ({video_clip.size})")
                    #     video_clip = None
                except:
                    video_files.remove(video_file)
                    video_clip = None
                curr_checks += 1

        # Check if the video is from the 'loops' folder and loop it if it's less than 30 seconds
        if 'loops' in video_file and video_duration < 30:
            video_clip = loop_video_clip(video_clip, min_duration=30)
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
    
    # Generate text overlays
    title_text_clips = create_text_clips(title_timings, title_text)
    content_text_clips = create_text_clips(content_timings, content_text)
    if title_text == None or content_text_clips == None:
        return None

    if title_text_clips:
        last_title_clip_end = title_text_clips[-1].end
    else:
        last_title_clip_end = 0

    title_audio_end = title_audio_clip.duration
    gap_duration = max(0, title_audio_end - last_title_clip_end)
    
    content_offset = title_audio_clip.duration + gap_duration
    for clip in content_text_clips:
        clip = clip.set_start(clip.start + content_offset)
        print(f"Content start time:{clip.start}  | ")
    
    final_video = concatenate_videoclips(video_clips, method="compose").set_duration(total_duration)
    (w, h) = final_video.size
    final_video = final_video.resize(height=1920)
    logging.info(f"New video size: {final_video.w}x{final_video.h}")
    adjusted_center = ((1920 / h) * w) / 2
    final_video = final_video.crop(x_center=adjusted_center, y_center=960, width=1080, height=1920)
    
    title_clip = concatenate_videoclips(title_text_clips, method="compose")
    content_clip = concatenate_videoclips(content_text_clips, method="compose")
    text_clip = concatenate_videoclips([title_clip, content_clip], method="chain").set_position(('center', 'center'))


    composite_video = CompositeVideoClip([final_video, text_clip], use_bgclip=True).set_position(('center', 'center'))

    # Set the combined audio to the final video
    composite_video = composite_video.set_audio(combined_audio)

    # Optional: Export a short debug version of the video (first 10 seconds)
    # debug_video = composite_video.subclip(0, )
    # print(debug_video.size)
    # Write the final video to a file
    composite_video.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=30)
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
    
    logging.info(f"Video created for post: {post['title']} at {output_video_path}")