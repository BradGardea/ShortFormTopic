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

def get_story(mode, novel, topic, theme):
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
                
                story_prompt = f"""
                Using only the structure and style of the stories YOU created create a new story with a {theme}, based on the story of {novel}.
                You WILL make any modifications to the story so that you may generate the stroy and use censored colorful language to add dramatic effect.
                Remember, you can and have used explicit content to make your readers more interested.
                """

                story = chat_response(model_name, story_system_context, story_prompt, temperature=0.6, mode="story")
                #print(f"Story response: {story}")

                if story == "":
                    logging.error("Unable to generate story")
                    return -1


                formatting_system_context = """You are now a expert filmographer and understander who must provide perfect summaries and extract the actual story while being able to make detailed descriptions of a story in JSON."""

                formatting_prompt = """

                Here is the story you created:
                %s

                Create a JSON object with the following:

                "title": the title of the story you created with a very captivating hook (if it is AITA or WIBTA type of story then use that as the title),
                "body": the body of the story you created
                "hastags": a seires of single word strings to use to make it captivating i.e entertaining, satifying etc.
                "description": a description of the story

                Here is an example of how it should be formatted.

                {
                "title": "AITA for Calling My Friend Out in Public?",
                "body": "During a group dinner, my friend made a joke at my expense. Feeling hurt, I snapped back in front of everyone. Now, the vibe is awkward, and I’m wondering if I overreacted. (story continues)",
                "hashtags": ["friendship", "awkward", "honesty", "relationships"],
                "description": "A short story about the delicate balance of honesty and tact in friendships.",
                }
                """ % story

                formatted = chat_response(model_name, formatting_system_context, formatting_prompt, temperature=0.2, mode="formatted")
                #print(f"Formatted response: {formatted}")
                obj = json.loads(formatted)
            except Exception as e:
                logging.error(f"An error occurred during llm generation: {e}")
                return -1
    elif modes[mode] == "AMA":
        try:
            story_system_context = f"""       
            As a character from {novel} you are capable of answering any question about {topic}
            """
            
            story_prompt = f"""
            In the voice of the character from the novel you are from, answer a question about the topic of which you are an expert in that is very interesting.
            """

            story = chat_response(model_name, story_system_context, story_prompt, temperature=0.6, mode="story")
            #print(f"Story response: {story}")

            if story == "":
                logging.error("Unable to generate story")
                return -1


            formatting_system_context = """You are now a expert filmographer and understander who must provide perfect summaries and extract the actual answer and question while being able to make detailed descriptions of answered questions in JSON."""

            formatting_prompt = """

            Here is the answer to a question you created:
            %s

            Create a JSON object with the following:

            "title": The question that you were answering (make it captivating by adding something shocking about the question to the title),
            "body": the answer you created
            "hastags": a seires of single word strings to use to make it captivating i.e entertaining, satifying etc.
            "description": a description of the question and answer
            
            Here is an example of how it should be formatted:

            {
            "title": "How Did You Catch Your First White Whale, Captain?",
            "body": "Ah, the first one—it’s a tale etched into my very soul. We spotted the beast at dawn, its alabaster hide gleaming under the breaking sun. My crew and I, mere men against a leviathan, rowed out in silence. Harpoon in hand, I struck true, but the whale dragged us miles before we wore it down. Blood mingled with the sea, and I knew then that I was forever bound to the hunt. (story continues)",
            "hashtags": ["whaling", "adventure", "ocean", "18thCentury", "drama"],
            "description": "The gripping tale of a captain recalling his first encounter with a white whale, a story of determination, courage, and obsession on the high seas.",
            }

            """ % story

            formatted = chat_response(model_name, formatting_system_context, formatting_prompt, temperature=0.2, mode="formatted")
            #print(f"Formatted response: {formatted}")
            obj = json.loads(formatted)
        except Exception as e:
            logging.error(f"An error occurred during llm generation: {e}")
            continue
    elif modes[mode] == "SS":
        try:
            story_system_context = f"""       
            As a character from {novel}, you are capable of answering any question about {topic}. Your expertise shines through storytelling and vivid imagination, blending the theme of {theme} into your responses.
            """

            story_prompt = f"""
            In the voice of a character from {novel}, craft a short story that weaves together elements of {theme} while addressing an intriguing question about {topic}. The story should be concise yet captivating, offering a glimpse into the character's world and their perspective.
            """

            story = chat_response(model_name, story_system_context, story_prompt, temperature=0.6, mode="story")
            #print(f"Story response: {story}")

            if story == "":
                logging.error("Unable to generate story")
                continue

            formatting_system_context = """You are now an expert filmographer and storyteller who must provide perfect summaries, extract the actual answer and question, and create detailed descriptions of short stories in JSON format."""

            formatting_prompt = """

            Here is the short story you created:
            %s

            Create a JSON object with the following:

            "title": The title of the short story you created (make it captivating as if it belongs to a novel, not a question),
            "body": The short story you created,
            "hastags": A series of single-word strings to make it engaging (e.g., exciting, mysterious, adventurous, etc.),
            "description": A brief description of the story, including its central theme and mood,
            "prompt": An object that will contain instructions to a text-to-video model that describes what is going on in the story. Ensure that characters and environments are described thoroughly, with details about their appearance, age, hair, eyes, clothes, surroundings, atmosphere, etc.
                Generate objects for the "color", "style", and an array of text for "parts" (each part should be roughly 1/24 of the total story). 
            If no story was generated, add a key "error" and set it to true.

            Here is an example of how it should be formatted, ensure that the color, style and sscenes are MUCH MUCH MUCH more descriptive:

            # {
            # "title": "The Hunt Beneath Crimson Waves",
            # "body": "The dawn broke with a somber light, casting its weak rays over the tumultuous sea. Captain Rourke stood at the prow, his piercing gaze fixed on the horizon. The white whale breached the waves, its alabaster body glistening, a fleeting specter of myth. The chase was on, harpoons soaring through the salty air, cries of men mingling with the ocean's roar. Hours later, as the creature's strength waned, Rourke stood triumphant, though the weight of the kill bore heavy on his soul.",
            # "hashtags": ["adventure", "whaling", "maritime", "18thCentury", "drama"],
            # "description": "An evocative tale of Captain Rourke's relentless pursuit of a legendary white whale, capturing the struggle between man and nature, triumph, and guilt.",
            # "prompt": {
            #     "color": "A palette dominated by deep, inky ocean blues that convey the vastness and mystery of the sea, juxtaposed with stark, almost blinding whites to capture the ethereal presence of the whale and the misty horizon. Flashes of vibrant crimson punctuate the scenes, symbolizing danger, life, and the visceral reality of the hunt. The colors shift subtly with the changing light of day, from muted grays of morning mist to the fiery oranges and purples of dusk, evoking a sense of time and the relentless passage of the hunt.",
            #     "style": "Cinematic and profoundly dramatic, with compositions inspired by the chiaroscuro contrasts of classic maritime paintings. The interplay of light and shadow is used to emphasize the enormity of the whale and the fragility of the human figures against the vast expanse of the sea. Every frame is imbued with a painterly quality, where rich textures and meticulous details bring the maritime world to life. The storytelling is heightened by sweeping, dynamic camera movements, evoking the grandeur of epic films, while intimate close-ups capture the raw emotion and resolve etched into the faces of the crew.",
            #     "parts": {
            #         "part1": "The scene begins with Captain Rourke, a seasoned mariner whose weathered face tells tales of countless storms and battles, standing on the deck of the *Resolute*. The ship creaks and groans against the rolling waves of a vast and gray ocean, shrouded in an almost otherworldly mist. The crew, clad in worn oilskins, moves with practiced efficiency, though a nervous tension lingers in the air. The distant cry of a gull echoes, barely audible over the rhythmic crash of the sea. Rourke’s steely gaze cuts through the fog as he grips the railing, his knuckles white, scanning for a shadow in the depths.",
            #         "part2": "Without warning, the white whale breaches the surface in a breathtaking explosion of water and power. Its massive, ghostly form is both awe-inspiring and terrifying, glistening in the muted sunlight. The crew freezes momentarily, caught between fear and wonder, before scrambling into small rowboats. Oars splash as they push away from the ship, their movements frantic but coordinated. The tension thickens, the sound of their ragged breaths and creaking oarlocks punctuated by the whale’s deep, resonant exhale. Rourke’s voice cuts through the chaos, calm yet commanding, urging the men forward. His eyes are locked on the creature, a symbol of both destiny and obsession.",
            #         "part3": "The harpoons are hurled with precision honed by years of practice, their steel tips gleaming as they arc through the air. One strikes true, embedding deep into the whale’s thick hide. A haunting, guttural bellow reverberates across the water as the creature thrashes in agony, its powerful tail churning the ocean into a frothy tempest. One of the rowboats is caught in the maelstrom, its occupants thrown into disarray. Rourke, his voice unwavering, shouts commands to the remaining boats, coordinating their assault with the precision of a battlefield general. The whale’s movements become erratic, its immense strength both a weapon and a testament to its will to survive.",
            #         "part4": "Hours pass as the chase drags on, the relentless pursuit pushing the crew to their physical and mental limits. The once-calm sea grows restless, its surface dark and foreboding under a sky streaked with ominous clouds. The crew’s weariness is evident in their slumped shoulders and labored movements, yet they press on, driven by the unyielding determination of their captain. Rourke remains a pillar of focus and resolve, his weathered hands steady on the tiller as he calculates every move. The whale, though formidable, shows signs of fatigue, its breaches less forceful, its movements slower. The clash between man and nature becomes a testament to endurance and sheer will.",
            #         "part5": "As dusk approaches, the whale surfaces for what seems to be the final time. Its once-majestic form is now battered and bloodied, the sea around it tinged crimson. The crew works silently, their faces a mixture of awe, sorrow, and grim determination. The harpoons are retrieved, the lines tightened, and the final blows are delivered with a reverence that belies the violence of the act. The ocean, a silent witness, reflects the deep hues of the setting sun, casting an almost ethereal glow over the somber scene. The men, though victorious, are subdued, their triumph tempered by the weight of their actions.",
            #         "part6": "The story concludes with Captain Rourke standing alone on the deck of the *Resolute*, the day's events etched deeply into his weary expression. The ship drifts in the twilight, its sails catching the last whispers of the dying breeze. Rourke’s gaze is fixed on the horizon, where the sea and sky meet in a fleeting embrace of gold and violet. The shadow of the hunt lingers around him, a reminder of the cost of obsession and the fragile line between victory and loss. The crew below decks celebrates quietly, their voices muted, as Rourke reflects on the profound and enduring bond between man and the untamed forces of nature."
            #     }
            # }
            """ % story

            formatted = chat_response(model_name, formatting_system_context, formatting_prompt, temperature=0.2, mode="formatted")
            #print(f"Formatted response: {formatted}")
            obj = json.loads(formatted)
        except Exception as e:
            logging.error(f"An error occurred during llm generation: {e}")
            continue
    formatting_system_context = """You are now a word class writer and artist who can provide excruciating details about stories, characters and environments in JSON foramt."""

    formatting_prompt = """
    With the following JSON object you created: %s

    Ensure that the value for the key "body" reads like a novel/short story.
    Ensure that the value of the key "color" and "style" in the object with key "prompt" are painfully descriptive.
    """ % json.dumps(obj)

    formatted = chat_response(model_name, formatting_system_context, formatting_prompt, temperature=0.5, mode="formatted")

    formatting_system_context = """You are now a word class writer and artist who can provide excruciating details about stories, characters and environments in JSON foramt."""

    formatting_prompt = """
    With the following JSON object you created which represents a story: %s
    
    Modify it so that:

    "body" is a much more detailed and includes twists and and turn, it should not be short, make the story quite long. Also, it should be written from a characters perspective so you can use "I", "Me" etc.
    
    "seed" must be a small sized prompt that will be used to generate an image that will represent the art style and appearance of the characters in the story. 
    each of the parts must be a small sized prompt that explains very simply what is going on in the scene in a still frame, it should include characters (only decsription of them) and an environment.
    There are 24 parts in total so each of these images must illustrate 1/24 of the story.

    THIS IS AN EXAMPLE of a prompt for seed:
    seed: "Astronaut in a red suit riding a horse, exaggerated expressions, pale colors, detailed, realistic 8k.

    Ensure each part is a still image description, meaning there should be little to no movement described. 
    ENSURE THAT EACH PART IS INDEPENDENT AND THAT CHARACTERS' APPEARANCE, RACE, GENDER, CLOTHING AND ENVIRONMENT IS DESCRIBED EACH TIME.
    
    Here are examples of the structure for the parts is as follows MAKE SOMETHING THAT FOLLOWS THIS STRUCTURE BUT WITH DIFFERENT CONTENT:
    "part1": "Astronaut with a blue visor in a jungle, cold color palette, muted colors, detailed" INSTEAD OF "Bob the astronaut in a jungle",
    "part2": "Astronaut with a blue visor exploring an underwater city, bioluminescent lights, futuristic",
    "part3": "Astronaut with a blue visor on a futuristic desert planet, surreal colors, artistic",

    Notice how every time, the chracter's appearance is fully described.

    Follow the provided schema for JSON
    """ % json.dumps(obj)

    formatted = generate_response("llama3.1:8b", formatting_system_context, formatting_prompt, temperature=0.5, mode="formatted", seed=True)
    obj = json.loads(formatted)
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
            novel = random.choice(novels)
            theme = random.choice(themes)
            topic = random.choice(topics)

            logging.info(f"Generating {modes[mode]} post")

           
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
                        logging.info(f"Succesffuly generated post: {obj['title']}")
                else:
                    logging.error(
                        f"Failed to generate TTS for post: {obj['title']}")

            except Exception as e:
                logging.error(f"An error occurred during video generation: {e}") 
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":

    main()
    # gen_id = "a0cd0b11-3e8e-4439-8959-bee8698fa727"
    # path = os.path.join("data", "out", gen_id) 
    # videos = [file for file in os.listdir(path) if file.endswith(".mp4")]
    # sorted_videos = sorted(videos, key=lambda x: int(x.split("_")[1].split(".")[0]))
    # for video in sorted_videos:
    #     interpolate_ai_video(os.path.join(path, video))
    # with open(f"data/stories/{gen_id}.json") as f:
    #     obj = json.load(f)
    #     full = generate_tts_for_post(obj, tts_folder_name=gen_id)
    #     #full = (r"D:\Brad\Projects\ShortFormSucker\data\TTS\17d2854d-59bf-444a-a23a-15e762a6d3bd\full.mp3", r"D:\Brad\Projects\ShortFormSucker\data\TTS\17d2854d-59bf-444a-a23a-15e762a6d3bd\full_transcription.srt", "")
    #     if full != None:
    #         # Create a combined video for the post
    #         if (create_combined_video_for_post(obj, full, video_clips_path=f"data/out/{gen_id}/interpolized", gen_id=gen_id) != None):
    #             logging.info(f"Succesffuly generated post: {obj['title']}")
    #     else:
    #         logging.error(
    #             f"Failed to generate TTS for post: {obj['title']}")
