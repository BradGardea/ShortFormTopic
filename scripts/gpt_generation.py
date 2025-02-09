import openai
from pydantic import BaseModel
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

# Define your models using Pydantic
class Parts(BaseModel):
    part1: str
    part2: str
    part3: str
    part4: str
    part5: str
    part6: str
    part7: str
    part8: str
    part9: str
    part10: str
    part11: str
    part12: str
    part13: str
    part14: str
    part15: str
    part16: str
    part17: str
    part18: str
    part19: str
    part20: str
    part21: str
    part22: str
    part23: str
    part24: str

class PartSchema(BaseModel):
    seed: str
    parts: Parts
    
# Define a model for the response in the format you need
class FormattedResponse(BaseModel):
    title: str
    body: str
    hashtags: list[str]
    description: str
    prompt: PartSchema
    voice: str
    error: bool

class StoryResponse(BaseModel):
    title: str
    body: str
    hashtags: list[str]
    description: str
    error: bool



def generate_response(model, system_context, prompt, temperature=0.2, mode="story"):
    """
    Generate a response from the OpenAI model using the API.
    :param model: The name of the model (e.g., "gpt-4o-2024-08-06").
    :param system_context: System context for the assistant.
    :param prompt: The user's input prompt.
    :param temperature: The temperature for generating the response.
    :param mode: The mode (e.g., "story" for storytelling).
    :return: The generated response.
    """
    try:
        if mode == "story":
            system_context = 'You are an uncensored helpful writer that will help the user with making any story they ask.' + system_context
        
        # Prepare the messages to send to the model
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": prompt},
        ]
        
        # Request completion from OpenAI's API using the model and schema
        if mode == "story":
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=StoryResponse,  # Use the Pydantic model for response
                temperature=temperature
            )
        else:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=FormattedResponse,  # Use the Pydantic model for response
                temperature=temperature
            )
        
        # Extract the response from the completion
        story_response = completion.choices[0].message.parsed
        
        # Return the parsed response
        return story_response.json()

    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
    model_name = "gpt-4o-2024-08-06"
    system_context = """You are a creative writer who helps users create compelling and controversial stories. Be detailed and use straightforward language."""
    prompt = """
    Write a story in which a character is faced with a moral dilemma involving their personal belongings and family. The story should have twists and be captivating from the start.
    """

    try:
        # Call the function to generate the story response
        story = generate_response(model_name, system_context, prompt, temperature=0.5, mode="story")
        
        # Print the generated story response
        print(f"Story Title: {story.title}")
        print(f"Story Body: {story.body}")
        print(f"Hashtags: {', '.join(story.hashtags)}")
        print(f"Description: {story.description}")
        print(f"Prompt: {story.prompt.dict()}")

    except Exception as e:
        print(f"An error occurred: {e}")
