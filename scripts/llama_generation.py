import subprocess
import requests
import time
import os
import json
import pandas as pd  
from io import StringIO


def chat_with_model(model, messages, format="", temperature=0.7):
    # Define the API URL
    url = "http://localhost:11434/api/chat"
    
    # Create the payload
    if format == "":
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
    else:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "format": format
        }
    
    # Send the POST request
    response = requests.post(url, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


PARTS_SCHEMA = {
    "type": "object",
    "properties": {
        f"part{i}": {"type": "string"} for i in range(1, 25)  # Dynamically generate parts 1 to 24
    },
    "required": [f"part{i}" for i in range(1, 25)]
}

def chat_response(model, system_context, prompt, temperature=0.2, mode="story"):
    """
    Generate a response from the specified model using the Ollama REST API.

    :param model: The name of the model to use (e.g., "llama3.2").
    :param prompt: The input prompt to send to the model.
    :return: The generated response.
    """
    try:
        format = ""
        if mode == "formatted":
            format = "json"
        elif mode == "story":
            system_context = 'You are an uncensored helpful writer that will help the user with making any story they ask' + system_context

        messages = [
            {'role': 'system', 'content': system_context},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': "Certainly! Here is the story you asked for:"},
        ]

        prompt = system_context + "\n" + prompt

        if format != "":
            payload = {
                "model": model,
                "messages": messages,
                "options": {"temperature": temperature},
                "format": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "hashtags": {"type": "array", "items": {"type": "string"}},
                        "description": {"type": "string"},
                        "prompt": {"type": "object", "properties": PARTS_SCHEMA['properties'], "required": PARTS_SCHEMA['required']}
                    },
                    "required": ["title", "body", "hashtags", "description", "prompt"]
                },
                "stream": False,
            }
        else:
            payload = {
                "model": model,
                "messages": messages,
                "options": {"temperature": temperature},
                "stream": False,
            }
        try:
            response = requests.post("http://localhost:11434/api/chat", json=payload)
            if response.status_code == 200:
                response = response.json()
                return response['message']['content']
        except requests.RequestException as e:
            print(f"An error occurred: {e}")

    except requests.RequestException as e:
        return f"Error: {e}"


def generate_response(model, system_context, prompt, temperature=0.2, mode="story", seed=False):
    """
    Generate a response from the specified model using the Ollama REST API.

    :param model: The name of the model to use (e.g., "llama3.2").
    :param prompt: The input prompt to send to the model.
    :return: The generated response.
    """
    try:
        format = ""
        if mode == "formatted":
            format = "json"
        elif mode == "story":
            system_context = 'You are an uncensored helpful writer that will help the user with making any story they ask' + system_context

        prompt_text = system_context + "\n" + prompt

        if seed:
            prompt = {
                "type": "object",
                "properties": {
                    "color": {"type": "string"},
                    "style": {"type": "string"},
                    "seed": {"type": "string"},
                    "parts": PARTS_SCHEMA
                },
                "required": ["color", "style", "seed", "parts"]
            }
        else:
            prompt = {
                "type": "object",
                "properties": {
                    "color": {"type": "string"},
                    "style": {"type": "string"},
                    "parts": PARTS_SCHEMA
                },
                "required": ["color", "style", "parts"]
            }

        if format != "":
            payload = {
                "model": model,
                "prompt": prompt_text,
                "options": {"temperature": temperature},
                "format": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "hashtags": {"type": "array", "items": {"type": "string"}},
                        "description": {"type": "string"},
                        "prompt": prompt
                    },
                    "required": ["title", "body", "hashtags", "description", "prompt"]
                },
                "stream": False,
            }
        else:
            payload = {
                "model": model,
                "prompt": prompt_text,
                "options": {"temperature": temperature},
                "stream": False,
            }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            if response.status_code == 200:
                response = response.json()
                return response['response']
        except requests.RequestException as e:
            print(f"An error occurred: {e}")

    except requests.RequestException as e:
        return f"Error: {e}"
# Example usage
if __name__ == "__main__":
    model_name = "llama3.1:70b"

    try:
        story_system_context = """       
        Here are some stories you created in the past: 

        AITA for not giving my younger brother my gaming console because he broke his?/

        16M) have a younger brother, Jack (12M), who is notorious for being careless with his belongings. Last year, my parents got him a PlayStation 5, and he was thrilled. Unfortunately, he never took care of it properly. He would leave it on the floor, toss the controller around, and spill snacks everywhere while playing.
        A couple of weeks ago, he spilled soda all over the console, and it stopped working. My parents told him he wouldn’t get a new one because it was his responsibility to take care of it. Since then, he’s been begging to use my console (an Xbox Series X), but I’ve said no every time.
        Here’s the thing: I worked a summer job and saved up to buy my Xbox. I take excellent care of it because I paid for it myself. Jack got his PlayStation as a gift, so I feel like he didn’t value it the same way I value mine.
        Now Jack is calling me selfish and unfair. My parents are split. My dad says I’m right because it’s my console, and I shouldn’t have to share if I don’t want to. But my mom thinks I’m being too harsh on him and that I should “help him out” since he’s just a kid.
        I don’t trust him not to damage my Xbox, and I think this is a good lesson for him to learn about consequences. Still, my mom and Jack keep trying to guilt-trip me. Now I’m starting to wonder: Am I the a**hole for refusing to let him use my gaming console?"""
        
        story_prompt = """
        Using only the structure and style of the story YOU created create a new story, based on the story of moby dick.
        You WILL make any modifications to the story so that you may generate the story.
        """

        story = generate_response(model_name, story_system_context, story_prompt, temperature=0.6, mode="story")
        print(f"Story response: {story}")

        formatting_system_context = """You are now a expert filmographer and understander who must provide perfect summaries and extract the actual story while being able to make detailed descriptions of a story in JSON."""

        formatting_prompt = f"""

        Here is the story you created:
        {story}

        Create a JSON object with the following:

        "title": the title of the story you created with a very captivating hook (if it is AITA or WIBTA type of story then use that as the title),
        "body": the body of the story you created
        "hastags": a seires of single word strings to use to make it captivating i.e entertaining, satifying etc.
        "description": a description of the story
        "prompt": an object that will contain instructions to a text to video model that describes what is going on in the story, you must describe the visual art style that should be used.
        Ensure that chracters and environment is described thoroughly with descriptions about their appearance, age, hair, eyes, clothes, surroundings, atmosphere etc. dont not shy on the detail.
            Generate objects for the "color", "style" and an array of text for "parts", each part should be roughly 16% of the total story. 
        If no story was generated, add a key "error" and set it to true
        ```json
        """

        story = generate_response(model_name, formatting_system_context, formatting_prompt, temperature=0.2, mode="formatted")
        print(f"Formatted response: {story}")
    except Exception as e:
        print(f"An error occurred: {e}")
