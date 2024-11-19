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

def loop_video_clip(video_clip, min_duration=30):
    """ Loop a video clip until it exceeds the specified duration. """
    video_duration = video_clip.duration
    if video_duration >= min_duration:
        return video_clip
    # Repeat the video clip until it exceeds min_duration
    loop_count = int(min_duration // video_duration) + 1
    return concatenate_videoclips([video_clip] * loop_count).subclip(0, min_duration)

audio = AudioFileClip("data/part2.mp3")

movieclip = VideoFileClip("data/thumbs_up_2.gif")
movieclip = loop_video_clip(movieclip, audio.duration)
movieclip = movieclip.set_audio(audio)

movieclip.write_videofile("data/like_part_2.mp4", codec="libx264", audio_codec="aac", fps=30)
