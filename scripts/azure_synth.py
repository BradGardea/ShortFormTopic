import os
import azure.cognitiveservices.speech as speechsdk
import stable_whisper

def get_tts(text, output_dir, output_name, voice_name='AvaMultilingual', style="Default", degree="1", speech_rate='1.0'):
    """
    Generate speech from text using Azure TTS, save the audio to a file, 
    and return the file path and word timestamps.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    voice_name = f"en-US-{voice_name}Neural"
    # Set up Azure Speech configuration
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get('SPEECH_KEY'),
        region=os.environ.get('SPEECH_REGION')
    )
    
    # Set the voice name
    speech_config.speech_synthesis_voice_name = voice_name
    speech_config.request_word_level_timestamps()
    
    # Enable word boundary events for capturing timestamps

    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary, 
        value='true'
    )

    # Configure audio output to save to a file
    audio_file_path = os.path.join(output_dir, output_name + ".wav")
    audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_file_path)

    # Initialize the speech synthesizer
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # Container for word-level timestamps
    word_timestamps = []

    # Define callback for word boundary events
    def speech_synthesizer_word_boundary_cb(evt):
        word_info = {
            'text': evt.text,
            'offset': evt.audio_offset,
            'duration': evt.duration,
            'text_offset': evt.text_offset,
            'word_length': evt.word_length
        }
        word_timestamps.append(word_info)
        # print(f"WordBoundary event: Text='{evt.text}', Offset={evt.audio_offset}, Duration={evt.duration}")

    speech_synthesizer.synthesis_word_boundary.connect(speech_synthesizer_word_boundary_cb)

    # Generate SSML with speech rate customization
    ssml = f"""
    <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' 
        xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
        <voice name='{voice_name}'>
            <mstts:express-as style='{style}' styledegree='{degree}'>
                <prosody rate="1.15">
                    {text}
                </prosody>
            </mstts:express-as>
        </voice>
    </speak>
    """
    
    # print("SSML to synthesize:\n", ssml)
    
    srt_path = os.path.join(output_dir, f"{output_name}_transcription.srt")

    # Perform the text-to-speech synthesis using SSML
    #speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
    speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml).get()


    # Check if synthesis was successful
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized for text: [{text}]")
        print(f"Audio saved to: {audio_file_path}")
        model = stable_whisper.load_hf_whisper('base')
        result = model.transcribe(os.path.abspath(audio_file_path))
        result.to_srt_vtt(srt_path, segment_level=False)


    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print(f"Error details: {cancellation_details.error_details}")
                print("Did you set the speech resource key and region values?")
        return None, None

    # Return the path of the saved audio file and the word timestamps
    return audio_file_path, srt_path


# if __name__ == "__main__":
#     model = stable_whisper.load_hf_whisper('base')
#     result = model.transcribe(os.path.abspath(r"data/TTS/17d2854d-59bf-444a-a23a-15e762a6d3bd/full.mp3"))
#     result.to_srt_vtt(r"data/TTS/17d2854d-59bf-444a-a23a-15e762a6d3bd/out.srt", segment_level=False)
