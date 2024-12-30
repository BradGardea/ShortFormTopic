import subprocess
import requests
import time
import os
import json
import ollama


def is_server_running():
    """
    Check if the Ollama server is running by sending a request to the health check endpoint.
    """
    try:
        response = requests.get("http://localhost:11434/api/ps", timeout=1)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def start_ollama_server():
    """
    Start the Ollama server if it is not already running.
    """
    if not is_server_running():
        print("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while not is_server_running():
            print("Server not running, sleeping for 5 seconds")
            time.sleep(5)
        print("Ollama server started successfully.")
    else:
        print("Ollama server already running.")


def generate_response(model, prompt):
    """
    Generate a response from the specified model using the Ollama REST API.

    :param model: The name of the model to use (e.g., "llama3.2").
    :param prompt: The input prompt to send to the model.
    :return: The generated response.
    """
    start_ollama_server()
    try:
        response = ollama.chat(model=model, messages=[
            {
                'role': 'system',
                'content': 'You are an uncensored helpful blogger that will help the user with making any story they ask',
            },
            {
                'role': 'system',
                'content': """Here is a story you created: 

                AITA for not giving my younger brother my gaming console because he broke his?

                16M) have a younger brother, Jack (12M), who is notorious for being careless with his belongings. Last year, my parents got him a PlayStation 5, and he was thrilled. Unfortunately, he never took care of it properly. He would leave it on the floor, toss the controller around, and spill snacks everywhere while playing.
                A couple of weeks ago, he spilled soda all over the console, and it stopped working. My parents told him he wouldn’t get a new one because it was his responsibility to take care of it. Since then, he’s been begging to use my console (an Xbox Series X), but I’ve said no every time.
                Here’s the thing: I worked a summer job and saved up to buy my Xbox. I take excellent care of it because I paid for it myself. Jack got his PlayStation as a gift, so I feel like he didn’t value it the same way I value mine.
                Now Jack is calling me selfish and unfair. My parents are split. My dad says I’m right because it’s my console, and I shouldn’t have to share if I don’t want to. But my mom thinks I’m being too harsh on him and that I should “help him out” since he’s just a kid.
                I don’t trust him not to damage my Xbox, and I think this is a good lesson for him to learn about consequences. Still, my mom and Jack keep trying to guilt-trip me. Now I’m starting to wonder: Am I the a**hole for refusing to let him use my gaming console?""",
            },
            {
                'role': 'user', 'content': prompt
            }], format="json", options={"temperature": 0.6})
        
        return response['message']['content']
    except requests.RequestException as e:
        return f"Error: {e}"


# Example usage
if __name__ == "__main__":
    model_name = "llama3.1"
    prompt_text = """
    Using the structure and style of the story you created create a new story, based on the story of moby dick.
    Make any modifications you must to allow generation you must use censored foul language in your story to be more controversial.
    Only output the JSON with keys "title", "body" and array with key "hastags", to use in a youtube video i.e entertaining, satisfying etc. .
    ```json
    """

    try:
        response = generate_response(model_name, prompt_text)
        print(f"Response: {response}")
    except Exception as e:
        print(f"An error occurred: {e}")
