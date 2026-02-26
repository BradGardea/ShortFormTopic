import os
import azure.cognitiveservices.speech as speechsdk
import stable_whisper
import re

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
                <prosody rate="1.1">
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


import os
import stable_whisper

def create_transcript(audio_path, output_dir=None, output_name=None, format="srt"):
    """
    Transcribes an audio file and saves the transcript in the specified format (SRT or VTT).
    
    Args:
        audio_path (str): Path to the input audio file.
        output_dir (str, optional): Directory to save the transcript. Defaults to same as audio_path.
        output_name (str, optional): Name for the output file (without extension). Defaults to audio filename.
        format (str): Output format: 'srt' or 'vtt'.

    Returns:
        str: Path to the saved transcript file.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load the Whisper model
    model = stable_whisper.load_model('base')

    # Transcribe the audio
    result = model.transcribe(audio_path)

    # Set default output values if not provided
    if output_dir is None:
        output_dir = os.path.dirname(audio_path)
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(audio_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    # Determine output file path
    extension = "srt" if format.lower() == "srt" else "vtt"
    output_path = os.path.join(output_dir, f"{output_name}_transcription.{extension}")

    # Save the transcription
    if format.lower() == "srt":
        result.to_srt_vtt(output_path, segment_level=False)
    elif format.lower() == "vtt":
        result.to_srt_vtt(output_path, vtt=True, segment_level=False)
    else:
        raise ValueError("Unsupported format. Use 'srt' or 'vtt'.")

    print(f"Transcript saved to: {output_path}")
    return output_path


def merge_srt_words_to_sentences(srt_path, output_txt_path=None):
    """
    Merge individual word-level SRT lines into full sentences.
    
    Args:
        srt_path (str): Path to the SRT file.
        output_txt_path (str, optional): Path to save the merged sentences. If None, just returns list.

    Returns:
        List[Dict]: Each dict has 'start', 'end', and 'sentence'.
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    sentences = []
    current_sentence = ""
    start_time = None
    end_time = None

    timecode_re = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})')

    i = 0
    while i < len(lines):
        # Skip sequence number line
        if lines[i].strip().isdigit():
            i += 1
            continue

        # Match timecode
        match = timecode_re.match(lines[i])
        if match:
            start = match.group(1)
            end = match.group(2)
            i += 1
            if i < len(lines):
                word = lines[i].strip()
                if not current_sentence:
                    start_time = start
                current_sentence += ("" if current_sentence == "" else " ") + word
                end_time = end

                # Check if sentence ends
                if re.search(r'[.?!]$', word):
                    sentences.append({
                        "start": start_time,
                        "end": end_time,
                        "sentence": current_sentence.strip()
                    })
                    current_sentence = ""
                    start_time = None
                    end_time = None
        i += 1

    # Add trailing sentence if no punctuation at the end
    if current_sentence:
        sentences.append({
            "start": start_time,
            "end": end_time,
            "sentence": current_sentence.strip()
        })

    # Optionally write to file
    if output_txt_path:
        with open(output_txt_path, 'w', encoding='utf-8') as out_f:
            for s in sentences:
                out_f.write(f"[{s['start']} --> {s['end']}] {s['sentence']}\n")

    return sentences


if __name__ == "__main__":
    #create_transcript("C://Users//brad8//Downloads//Robinhood's_CEO_on_the_Plan_to_Tokenize_Everything.mp3", output_dir="transcripts", format="srt")
    sents = merge_srt_words_to_sentences(r"C:\B\Work\ShortFormSucker\transcripts\Robinhood's_CEO_on_the_Plan_to_Tokenize_Everything_transcription.srt", output_txt_path="transcripts//merged_sentences.txt")



import os
import stable_whisper

def create_transcript(audio_path, output_dir=None, output_name=None, format="srt"):
    """
    Transcribes an audio file and saves the transcript in the specified format (SRT or VTT).
    
    Args:
        audio_path (str): Path to the input audio file.
        output_dir (str, optional): Directory to save the transcript. Defaults to same as audio_path.
        output_name (str, optional): Name for the output file (without extension). Defaults to audio filename.
        format (str): Output format: 'srt' or 'vtt'.

    Returns:
        str: Path to the saved transcript file.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load the Whisper model
    model = stable_whisper.load_model('base')

    # Transcribe the audio
    result = model.transcribe(audio_path)

    # Set default output values if not provided
    if output_dir is None:
        output_dir = os.path.dirname(audio_path)
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(audio_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    # Determine output file path
    extension = "srt" if format.lower() == "srt" else "vtt"
    output_path = os.path.join(output_dir, f"{output_name}_transcription.{extension}")

    # Save the transcription
    if format.lower() == "srt":
        result.to_srt_vtt(output_path, segment_level=False)
    elif format.lower() == "vtt":
        result.to_srt_vtt(output_path, vtt=True, segment_level=False)
    else:
        raise ValueError("Unsupported format. Use 'srt' or 'vtt'.")

    print(f"Transcript saved to: {output_path}")
    return output_path


def merge_srt_words_to_sentences(srt_path, output_txt_path=None):
    """
    Merge individual word-level SRT lines into full sentences.
    
    Args:
        srt_path (str): Path to the SRT file.
        output_txt_path (str, optional): Path to save the merged sentences. If None, just returns list.

    Returns:
        List[Dict]: Each dict has 'start', 'end', and 'sentence'.
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    sentences = []
    current_sentence = ""
    start_time = None
    end_time = None

    timecode_re = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})')

    i = 0
    while i < len(lines):
        # Skip sequence number line
        if lines[i].strip().isdigit():
            i += 1
            continue

        # Match timecode
        match = timecode_re.match(lines[i])
        if match:
            start = match.group(1)
            end = match.group(2)
            i += 1
            if i < len(lines):
                word = lines[i].strip()
                if not current_sentence:
                    start_time = start
                current_sentence += ("" if current_sentence == "" else " ") + word
                end_time = end

                # Check if sentence ends
                if re.search(r'[.?!]$', word):
                    sentences.append({
                        "start": start_time,
                        "end": end_time,
                        "sentence": current_sentence.strip()
                    })
                    current_sentence = ""
                    start_time = None
                    end_time = None
        i += 1

    # Add trailing sentence if no punctuation at the end
    if current_sentence:
        sentences.append({
            "start": start_time,
            "end": end_time,
            "sentence": current_sentence.strip()
        })

    # Optionally write to file
    if output_txt_path:
        with open(output_txt_path, 'w', encoding='utf-8') as out_f:
            for s in sentences:
                out_f.write(f"[{s['start']} --> {s['end']}] {s['sentence']}\n")

    return sentences


if __name__ == "__main__":
    #create_transcript("C://Users//brad8//Downloads//Robinhood's_CEO_on_the_Plan_to_Tokenize_Everything.mp3", output_dir="transcripts", format="srt")
    sents = merge_srt_words_to_sentences(r"C:\B\Work\ShortFormSucker\transcripts\Robinhood's_CEO_on_the_Plan_to_Tokenize_Everything_transcription.srt", output_txt_path="transcripts//merged_sentences.txt")

