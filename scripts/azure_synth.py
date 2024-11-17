import os
import azure.cognitiveservices.speech as speechsdk

def get_tts(text, output_dir, output_name):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up Azure Speech configuration
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get('SPEECH_KEY'),
        region=os.environ.get('SPEECH_REGION')
    )
    
    # Set voice to neural multilingual
    speech_config.speech_synthesis_voice_name = 'en-US-AvaMultilingualNeural'
    
    # Configure audio output to save to a file
    audio_file_path = os.path.join(output_dir, output_name)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_file_path)
    
    # Initialize the speech synthesizer with audio configuration
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )
    
    # Container for word-level timestamps
    word_timestamps = []

    # Define callback for word boundary events to capture timestamps
    def word_boundary_handler(evt):
        word_info = {
            'text': evt.text,
            'offset': evt.audio_offset / 10_000,  # Convert to milliseconds
            'duration': evt.duration / 10_000  # Convert to milliseconds
        }
        word_timestamps.append(word_info)

    # Connect the event handler
    speech_synthesizer.synthesis_word_boundary.connect(word_boundary_handler)

    # Perform the text-to-speech synthesis
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    # Check if synthesis was successful
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized for text: [{text}]")
        print(f"Audio saved to: {audio_file_path}")
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print(f"Error details: {cancellation_details.error_details}")
                print("Did you set the speech resource key and region values?")
        return None, None

    # Return the path of the saved audio file and the word timestamps
    return audio_file_path, word_timestamps