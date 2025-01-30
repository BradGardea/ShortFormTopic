import stable_whisper
import os    


audio_file_path = r"C:\temp\test.wav"
srt_path = r"C:\temp\test.srt"


model = stable_whisper.load_hf_whisper('base')
result = model.transcribe(os.path.abspath(audio_file_path))
result.to_srt_vtt(srt_path, segment_level=False)