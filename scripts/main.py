import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
sys.path.insert(0, dname)  # Add parent directory to Python module search path

os.chdir(dname)

print("Current working dir", os.getcwd())
import random
import logging
import time
import threading
import pandas as pd
import uuid
import json

from scripts.reddit import run_scrapers
from scripts.azure_synth import get_tts
from scripts.video_generator import create_combined_video_for_post
from scripts.llama_generation import generate_response, chat_response
from scripts.ai_video_creation import generate_ai_video_stable_diffusion, interpolate_ai_video





modes = {
    0: "AITA",
    1: "AMA",
    2: "SS", # short story
}

model_name = "llama3.1:8b"

novels = [
  "Pride and Prejudice",
  "Sense and Sensibility",
  "Emma",
  "Northanger Abbey",
  "Mansfield Park",
  "Persuasion",
  "Twenty Thousand Leagues Under the Sea",
  "Around the World in Eighty Days",
  "Journey to the Center of the Earth",
  "The Mysterious Island",
  "From the Earth to the Moon",
  "Moby-Dick; or, The Whale",
  "Typee",
  "Omoo",
  "Billy Budd, Sailor",
  "The War of the Worlds",
  "The Time Machine",
  "The Invisible Man",
  "The Island of Doctor Moreau",
  "The First Men in the Moon",
  "Frankenstein; or, The Modern Prometheus",
  "Treasure Island",
  "The Strange Case of Dr. Jekyll and Mr. Hyde",
  "Jane Eyre",
  "Wuthering Heights",
  "Great Expectations",
  "A Tale of Two Cities",
  "David Copperfield",
  "The Adventures of Tom Sawyer",
  "Adventures of Huckleberry Finn",
  "The Tell-Tale Heart",
  "The Fall of the House of Usher",
  "The Raven",
  "Dracula",
  "The Scarlet Letter",
  "The House of the Seven Gables",
  "A Christmas Carol in Prose; Being a Ghost Story of Christmas",
  "Alice's Adventures in Wonderland",
  "The Adventures of Sherlock Holmes",
  "Middlemarch",
  "Little Women; Or, Meg, Jo, Beth, and Amy",
  "Crime and Punishment",
  "War and Peace",
  "The Brothers Karamazov",
  "Don Quixote",
  "Ulysses",
  "The Count of Monte Cristo"
]

themes = [
  "Action",
  "Adventure",
  "Suspense",
  "Horror",
  "Romance",
  "Mystery",
  "Science Fiction",
  "Fantasy",
  "Historical",
  "Drama",
  "Comedy",
  "Tragedy",
  "Gothic",
  "Psychological",
  "Thriller",
  "Epic",
  "Political",
  "Philosophical",
  "Satire",
  "Social Commentary",
  "Morality",
  "Survival",
  "Revenge",
  "Love",
  "Family",
  "Friendship",
  "War",
  "Freedom",
  "Exploration"
]

topics = [
  "Adultery",
  "Conflict",
  "Relationships",
  "Family",
  "Friendship",
  "Workplace",
  "School",
  "Ethics",
  "Parenting",
  "Health",
  "Fitness",
  "Diet",
  "Money",
  "Investments",
  "Debt",
  "Mental Health",
  "Pets",
  "Hobbies",
  "Travel",
  "Cooking",
  "Technology",
  "Cars",
  "Education",
  "Career Advice",
  "Legal Issues",
  "Fashion",
  "Shopping",
  "Housing",
  "Renting",
  "Home Improvement",
  "Dating",
  "Marriage",
  "Divorce",
  "Self-Improvement",
  "Spirituality",
  "Gaming",
  "Entertainment",
  "Books",
  "Movies",
  "Music",
  "Art",
  "Politics",
  "Culture",
  "History",
  "Science",
  "Environment",
  "Climate Change",
  "Social Media",
  "Conflict Resolution",
  "Etiquette",
  "Events",
  "Life Choices"
]

semantics = [
    "Triste",
    "Controversial",
    "Dreamy",
    "Exciting",
    "Riveting",
    "Touching",
    "Inspirational",
    "Mysterious",
    "Heartwarming",
    "Thrilling",
    "Melancholic",
    "Humorous",
    "Dramatic",
    "Chilling",
    "Romantic",
    "Hopeful",
    "Poignant",
    "Suspenseful",
    "Whimsical",
    "Dark",
    "Empowering",
    "Adventurous",
    "Bittersweet",
    "Epic",
    "Philosophical",
    "Haunting",
    "Euphoric",
    "Nostalgic",
    "Gripping",
]


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_random_stories_from_csv(filename='data/reddit_posts.csv', num_posts=3):
    """Pick a specified number of random stories that haven't been marked as 'generated'."""
    df = pd.read_csv(filename)
    # Filter out stories that have already been generated
    unprocessed_posts = df[(df['post_type'] != 'generated')
                           & (df['post_type'] != 'video')]

    # If there are no unprocessed posts, return an empty list
    if unprocessed_posts.empty:
        return []

    # Sample up to 'num_posts' random posts (or fewer if not enough available)
    return unprocessed_posts.sample(n=min(num_posts, len(unprocessed_posts))).to_dict(orient='records')


def generate_tts_for_post(post, tts_folder_name):
    """Generate TTS for the title and content of the post using Azure TTS."""
    title = post['title'] + ". "
    content = post['body'] + "."

    #DEBUG
    content = content

    try:
        target_text = title + content
        full_path, timings_full = get_tts(
            target_text, f"data/TTS/{tts_folder_name}", "full")

        logging.info(f"TTS generation successful for post: {title}")
        return (full_path, timings_full, target_text)
    except Exception as e:
        logging.error(f"Error generating TTS for post {title}: {e}")
        return None


def update_post_as_generated(post_id, filename='data/reddit_posts.csv'):
    """Update the post as 'generated' in the CSV."""
    df = pd.read_csv(filename)
    df.loc[df['id'] == post_id, 'post_type'] = 'generated'
    df.to_csv(filename, index=False)
    logging.info(f"Marked post with ID {post_id} as 'generated'.")


def sleep_for_duration(seconds):
    """Function to sleep in a non-blocking manner using threading."""
    time.sleep(seconds)



def write_json_to_folder(json_object, folder_path, filename):
    """
    Write a JSON object to a file in the specified folder.

    Parameters:
    - json_object: The Python dictionary or list to write to a JSON file.
    - folder_path: The target folder where the JSON file should be saved.
    - filename: The name of the JSON file (e.g., 'data.json').

    Returns:
    - The full path to the saved JSON file.
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Full file path
    file_path = os.path.join(folder_path, filename)

    # Write the JSON object to the file
    with open(file_path, 'w') as json_file:
        json.dump(json_object, json_file, indent=4)

    logging.info(f"JSON file saved to: {file_path}")
    return file_path

def get_story(mode):

    use_novel = random.choice((True, False))
    novel = random.choice(novels)
    theme = random.choice(themes)
    topic = random.choice(topics)
    semantic = random.choice(semantics)
    #region llama
    if modes[mode] == "AITA":
        posts = get_random_stories_from_csv()
        stories = ""

        for post in posts:
            if post is not None:
                logging.info(
                    f"Selected post: {post['title']} (ID: {post['id']})")
                stories += post['title'] + "\n" + post['content'] + "\n"

            else:
                logging.info("Bad post, skipping")
        
        if stories == "":
            logging.error("Unable to generate story from database")
            return -1
        else:
            try:
                story_system_context = f"""       
                Here are some stories you created in the past: 

                {stories}
                """
                if use_novel:
                    story_prompt = f"""
                    Using only the structure and style of the stories YOU created create a new story with a {theme} theme, based on the story of {novel}.
                    
                    The story should be written to be {semantic} and captivating with a VERY interesting title and first line to captiavte the reader, use vulgar language and expressions. Ensure the material does not violate any laws.
                    """
                else:
                    story_prompt = f"""
                    Using only the structure and style of the stories YOU created create a new story with a {theme} theme.

                    The story should be written to be {semantic} and captivating with a VERY interesting title and first line, use vulgar language and expressions. Ensure the material does not violate any laws.
                    """

                obj = chat_response(model_name, story_system_context, story_prompt, temperature=0.6, mode="story")
                #print(f"Story response: {story}")

                if obj == "":
                    logging.error("Unable to generate story")
                    return -1

            except Exception as e:
                logging.error(f"An error occurred during llm generation: {e}")
                return -1
    elif modes[mode] == "AMA":
        try:
            story_system_context = f"""       
            You are now an expert about the novel {novel}, you are capable of answering any question about {topic}. Your expertise shines through storytelling and vivid imagination, blending the theme of {theme} into your responses.
            """
            
            story_prompt = f"""
            In the voice of the character from the novel you are from, create and answer a question about the topic of which you are an expert in that should be written to be {semantic} and interesting. Use vulgar language and expressions. Ensure the material does not violate any laws.
            """

            obj = chat_response(model_name, story_system_context, story_prompt, temperature=0.6, mode="story")
            #print(f"Story response: {story}")

            if obj == "":
                logging.error("Unable to generate story")
                return -1
        except Exception as e:
            logging.error(f"An error occurred during llm generation: {e}")
            return -1
    elif modes[mode] == "SS":
        try:
            story_system_context = f"""       
            You are now an expert about the novel {novel}, you can make any story on the topic of: {topic}. Your expertise shines through storytelling and vivid imagination, blending the theme of {theme} into your responses.
            """

            story_prompt = f"""
            In the voice of a character from {novel}, craft a short story that weaves together elements of {theme} while following {topic}. The story should be somewhat lengthy, captivating and should be written to be {semantic} with a VERY intriguing and contentious, the first line should act as a hook for the reader, offering a glimpse into the character's world and their perspective. Use vulgar language and expressions. Ensure the material does not violate any laws.
            """

            obj = chat_response(model_name, story_system_context, story_prompt, temperature=0.6, mode="story")
            #print(f"Story response: {story}")

            if obj == "":
                logging.error("Unable to generate story")
                return -1
            
        except Exception as e:
            logging.error(f"An error occurred during llm generation: {e}")
            return -1


    formatting_system_context = """You are now a word class writer and artist who can provide excruciating details about stories, characters and environments in JSON foramt."""

    formatting_prompt = """
    With the following story you created: %s
    
    Ensure that the value for the key "body" reads like a novel/short story.

    Ensure the story does not violate any laws. If it does, set "error" to true.
    """ % obj

    formatted = chat_response("llama:8b", formatting_system_context, formatting_prompt, temperature=0.5, mode="formatted")

    obj = json.loads(formatted)
    if obj.get("error", False) == True:
        return -1

    formatting_system_context = """You are now a word class writer and artist who can provide excruciating details about stories, characters and environments in JSON foramt."""

    formatting_prompt = """
    With the following JSON object you created which represents a story: %s
    
    Modify it so that:
    "title" is the title of the story, it can also be question (if the story is an AITA story or an AMA story)    
    "seed" must be a small sized prompt that will be used to generate an image that will represent the art style and appearance of the characters in the story. 
    each of the parts must be a small sized prompt that explains very simply what is going on in the scene in a still frame, it should include characters (only decsription of them) and an environment.
    There are 24 parts in total so each of these images must illustrate 1/24 of the story.
    "style" is the art style to use
    "color" is the color pallate of the story

    THIS IS AN EXAMPLE of a prompt for seed:
    seed: "Astronaut in a red suit riding a horse, exaggerated expressions, pale colors, detailed, realistic 8k.

    Ensure each part is a still image description, meaning there should be little to no movement described. 
    ENSURE THAT EACH PART IS INDEPENDENT AND THAT CHARACTERS' APPEARANCE, RACE, GENDER, CLOTHING AND ENVIRONMENT IS DESCRIBED EACH TIME.
    
    Here are examples of the structure for the parts is as follows MAKE SOMETHING THAT FOLLOWS THIS STRUCTURE BUT WITH DIFFERENT CONTENT:
    "part1": "Astronaut with a blue visor in a jungle, cold color palette, muted colors, detailed" INSTEAD OF "Bob the astronaut in a jungle",
    "part2": "Astronaut with a blue visor exploring an underwater city, bioluminescent lights, futuristic",
    "part3": "Astronaut with a blue visor on a futuristic desert planet, surreal colors, artistic",

    Notice how every time, the chracter's appearance is fully described and the character is not referenced by their name, only their appearance each time.

    Follow the provided schema for JSON
    """ % formatted

    formatted = generate_response("llama3.1:8b", formatting_system_context, formatting_prompt, temperature=0.5, mode="formatted", seed=True)
    obj = json.loads(formatted)
    return obj
    #endregion



def main(title = None, content = None):
    """Main function to run the scraping, TTS, and video generation cycle."""
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(os.path.dirname(abspath))
    os.chdir(dname)


    while True:
        obj = {}
        interval = 30 
        # Step 1: Run scrapers in parallel for the specified interval
        # run_scrapers(interval)

        # Step 2: Process posts after scrapers are done
        # logging.info("Scraping completed. Now selecting and processing posts.")

        try:    
            mode = random.choice(list(modes.keys()))

            logging.info(f"Generating {modes[mode]} post")

            obj = get_story(mode)
            if isinstance(obj, int) and obj < 0:
                continue

            gen_id = str(uuid.uuid4())
            if len(obj) == 0:
                logging.error("Unable to extrat json story")
                continue
            write_json_to_folder(obj, "data/stories", gen_id + ".json")

            try:
                if generate_ai_video_stable_diffusion(obj, gen_id) < 0:
                    continue
                path = os.path.join("data", "out", gen_id ) 
                videos = [file for file in os.listdir(path) if file.endswith(".mp4")]
                for video in videos:
                    interpolate_ai_video(os.path.join(path, video))

                full = generate_tts_for_post(obj, tts_folder_name=gen_id)

                if full != None:
                    # Create a combined video for the post
                    if (create_combined_video_for_post(obj, full, video_clips_path=f"data/out/{gen_id}/interpolized", gen_id=gen_id) != None):
                        with open(f"out/{gen_id}/story.json", 'w') as f:
                            json.dump(obj, f)
                            print("Added object to out dir")
                        logging.info(f"Succesffuly generated post: {obj['title']}")   
                else:
                    logging.error(
                        f"Failed to generate TTS for post: {obj['title']}")

            except Exception as e:
                logging.error(f"An error occurred during video generation: {e}") 
        except Exception as e:
            print(f"An error occurred: {e}")

def create_custom(gen_id):
    path = os.path.join("data", "out", gen_id) 
    videos = [file for file in os.listdir(path) if file.endswith(".mp4")]
    sorted_videos = sorted(videos, key=lambda x: int(x.split("_")[1].split(".")[0]))
    # for video in sorted_videos:
    #     interpolate_ai_video(os.path.join(path, video))
    with open(f"data/stories/{gen_id}.json") as f:
        obj = json.load(f)
        # full = generate_tts_for_post(obj, tts_folder_name=gen_id)
        full = (f"D:\Brad\Projects\ShortFormSucker\data\TTS\{gen_id}\\full.mp3", f"D:\Brad\Projects\ShortFormSucker\data\TTS\{gen_id}\\full_transcription.srt", "")
        if full != None:
            # Create a combined video for the post
            if (create_combined_video_for_post(obj, full, video_clips_path=f"data/out/{gen_id}/interpolized", gen_id=gen_id) != None):
                logging.info(f"Succesffuly generated post: {obj['title']}")
        else:
            logging.error(
                f"Failed to generate TTS for post: {obj['title']}")
def create_custom(gen_id):
    path = os.path.join("data", "out", gen_id) 
    videos = [file for file in os.listdir(path) if file.endswith(".mp4")]
    sorted_videos = sorted(videos, key=lambda x: int(x.split("_")[1].split(".")[0]))
    # for video in sorted_videos:
    #     interpolate_ai_video(os.path.join(path, video))
    with open(f"data/stories/{gen_id}.json") as f:
        obj = json.load(f)
        # full = generate_tts_for_post(obj, tts_folder_name=gen_id)
        full = (f"D:\Brad\Projects\ShortFormSucker\data\TTS\{gen_id}\\full.mp3", f"D:\Brad\Projects\ShortFormSucker\data\TTS\{gen_id}\\full_transcription.srt", "")
        if full != None:
            # Create a combined video for the post
            if (create_combined_video_for_post(obj, full, video_clips_path=f"data/out/{gen_id}/interpolized", gen_id=gen_id) != None):
                logging.info(f"Succesffuly generated post: {obj['title']}")
        else:
            logging.error(
                f"Failed to generate TTS for post: {obj['title']}")

if __name__ == "__main__":

    main()
    # gen_id = "88309a9c-7bc1-4822-9713-0c5d6e50163e"
    # create_custom(gen_id)

