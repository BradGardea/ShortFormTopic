from moviepy.editor import *
from moviepy.config import change_settings


final_video = VideoFileClip('data/satisfying_videos\\regular\\a_perfect_scoop_of_kinetic_sand.mp4')

(w, h) = final_video.size
print(f"Original video size: {w} x {h}")
final_video = final_video.resize(height=1920)
print(f"New video size: {final_video.w}x{final_video.h}")
adjusted_center = ((1920 / h) * w) / 2
final_video = final_video.crop(x_center=adjusted_center, y_center=960, width=1080, height=1920)

text = "FOOBAR"
start_time = 0  # Convert to seconds
word_duration = 5
print(f"Adding text: {text} from {start_time} to {start_time + word_duration}")
change_settings({"IMAGEMAGICK_BINARY": "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
# Create a text clip that appears at the center of the screen
text_clip = TextClip(text, fontsize=200, color='black', font="TooneyNoodleNF", method="caption", align="center")
text_clip = text_clip.set_start(start_time)
text_clip = text_clip.set_position(('center', 'center'))
text_clip = text_clip.set_duration(word_duration)

composite_video = CompositeVideoClip([final_video, text_clip] ,use_bgclip=True).set_position(('center', 'center'))




composite_video.write_videofile("out/scaled.mp4", codec="libx264", audio_codec="aac", fps=10)