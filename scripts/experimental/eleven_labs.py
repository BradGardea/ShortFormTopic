import requests
import os
import json
import base64

def get_tts(text, output_dir, output_name):
    url = "https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB/with-timestamps"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
    }

    data = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        print(f"Error encountered, status: {response.status_code}, "
            f"content: {response.text}")
        quit()

    json_string = response.content.decode("utf-8")
    response_dict = json.loads(json_string)
    audio_bytes = base64.b64decode(response_dict["audio_base64"])

    os.makedirs(output_dir, exist_ok=True)

    out = os.path.join(output_dir, output_name)


    with open(out, 'wb') as f:
        f.write(audio_bytes)

    return response_dict['alignment']

def get_timings(data):
    characters = data['characters']
    start_times = data['character_start_times_seconds']
    end_times = data['character_end_times_seconds']

    # Step 1: Group characters into words based on non-alphanumeric checks
    words = []
    current_word = []
    word_start_time = None

    for i in range(len(characters)):
        # If starting a new word, capture the start time
        if not current_word:
            word_start_time = start_times[i]
        
        # Add the current character to the current word
        current_word.append(characters[i])
        
        # Check if this is not the last character and if the next character is non-alphanumeric
        if i < len(characters) - 1:
            next_char = characters[i + 1]
            curr_char = characters[i]
            if curr_char.isspace():
                word_end_time = end_times[i]
                words.append({
                    'word': ''.join(current_word),
                    'start_time': word_start_time,
                    'end_time': word_end_time
                })
                current_word = []
            if next_char.isspace():
                # If the next character is non-alphanumeric, finalize the current word
                word_end_time = end_times[i]
                words.append({
                    'word': ''.join(current_word),
                    'start_time': word_start_time,
                    'end_time': word_end_time
                })
                current_word = []
        
        # If it's the last character, finalize the current word
        elif i == len(characters) - 1:
            word_end_time = end_times[i]
            words.append({
                'word': ''.join(current_word),
                'start_time': word_start_time,
                'end_time': word_end_time
            })

    # Step 2: Print the resulting words with their start and end times
    # for word_info in words:
    #     print(f"Word: {word_info['word']}, Start Time: {word_info['start_time']}, End Time: {word_info['end_time']}")

    return words