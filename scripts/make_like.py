from moviepy.editor import *
import os
import sys
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
sys.path.insert(0, dname)  # Add parent directory to Python module search path

os.chdir(dname)

from scripts.azure_synth import get_tts

# def loop_video_clip(video_clip, min_duration=30):
#     """ Loop a video clip until it exceeds the specified duration. """
#     video_duration = video_clip.duration
#     if video_duration >= min_duration:
#         return video_clip
#     # Repeat the video clip until it exceeds min_duration
#     loop_count = int(min_duration // video_duration) + 1
#     return concatenate_videoclips([video_clip] * loop_count).subclip(0, min_duration)

# audio = AudioFileClip("data/part2.mp3")

# movieclip = VideoFileClip("data/thumbs_up_2.gif")
# movieclip = loop_video_clip(movieclip, audio.duration)
# movieclip = movieclip.set_audio(audio)

# movieclip.write_videofile("data/like_part_2.mp4", codec="libx264", audio_codec="aac", fps=30)

text = "Check below for part 2! If you enjoyed this kind of content, please like and subscribe for more, it really helps us out. THANK YOU for watching!!"

audio_path, _ = get_tts(text=text, output_dir="data", output_name="part2")

# audio = AudioFileClip(r"data\part2.wav")
audio = AudioFileClip(audio_path)


movieclip = VideoFileClip(r"data\LikeAndSubscribe.gif")
x1, x2 = 0, movieclip.w  # Keep full width
y1, y2 = 55, movieclip.h  # Adjust 'y1' based on the white space height

# Apply cropping
cropped_video = movieclip.crop(x1=x1, x2=x2, y1=y1, y2=y2)

def generate_silence(duration, fps=44100):
    """Generate a silent audio clip of a given duration."""
    return AudioClip(lambda t: np.zeros((np.atleast_1d(t).shape[0], 2)), duration=duration, fps=fps)

video_duration = cropped_video.duration
audio_duration = audio.duration if audio else 0

if audio_duration < video_duration:
    silence_duration = video_duration - audio_duration
    silence = generate_silence(silence_duration)
    extended_audio = concatenate_audioclips([audio, silence])
else:
    extended_audio = audio.set_duration(video_duration)

# Set the new audio to the video
cropped_video = cropped_video.set_audio(extended_audio)

# Write the final video file
cropped_video.write_videofile("data/like_part_2.mp4", codec="libx264", fps=90)