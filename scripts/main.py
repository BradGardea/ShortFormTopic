import random
import logging
import time
import threading
from reddit import run_scrapers
from azure_synth import get_tts
from video_generator import create_combined_video_for_post
import pandas as pd
import os
import uuid
from llama_generation import generate_response, chat_response
from ai_video_creation import generate_ai_video, interpolate_ai_video
import json




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
    title = post['title']
    content = post['bdoy']

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

def main(title = None, content = None):
    """Main function to run the scraping, TTS, and video generation cycle."""
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(os.path.dirname(abspath))
    os.chdir(dname)


    while True:
        obj = {}
        interval = 30 
        if content != None:
            post = {"title": title, "id": uuid.uuid4(), "content": content }

            logging.info(f"Selected post: {post['title']} (ID: {post['id']})")

            tts_folder_name = post['id']
            full = generate_tts_for_post(
                post, tts_folder_name=tts_folder_name)

            if full != None:
                if (create_combined_video_for_post(post, full) != None):
                    logging.info("Exiting")
            else:
                logging.error(
                    f"Failed to generate TTS for post: {post['title']}")
        else:
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
                        continue
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
                                continue


                            formatting_system_context = """You are now a expert filmographer and understander who must provide perfect summaries and extract the actual story while being able to make detailed descriptions of a story in JSON."""

                            formatting_prompt = """

                            Here is the story you created:
                            %s

                            Create a JSON object with the following:

                            "title": the title of the story you created with a very captivating hook (if it is AITA or WIBTA type of story then use that as the title),
                            "body": the body of the story you created
                            "hastags": a seires of single word strings to use to make it captivating i.e entertaining, satifying etc.
                            "description": a description of the story
                            "prompt": an object that will contain instructions to a text to video model that describes what is going on in the story, you must describe the visual art style that should be used.
                            Ensure that chracters and environment is described thoroughly with descriptions about their appearance, age, hair, eyes, clothes, surroundings, atmosphere etc. dont not shy on the detail.
                                Generate objects for the "color", "style" and an array of text for "parts", each part should be roughly 1/6 of the total story. 
                            If no story was generated, add a key "error" and set it to true

                            Here is an example of how it should be formatted, ensure that the color, style and sscenes are MUCH MUCH MUCH more descriptive:

                            {
                            "title": "AITA for Calling My Friend Out in Public?",
                            "body": "During a group dinner, my friend made a joke at my expense. Feeling hurt, I snapped back in front of everyone. Now, the vibe is awkward, and I’m wondering if I overreacted.",
                            "hashtags": ["friendship", "awkward", "honesty", "relationships"],
                            "description": "A short story about the delicate balance of honesty and tact in friendships.",
                            "prompt": {
                                "color": "The palette embraces neutral tones, with soft beige and muted grays dominating the background, lending a subdued and intimate ambiance to the restaurant scene. Warm, golden hues from overhead lighting create cozy pockets of light that highlight the characters' faces, drawing attention to their emotions and expressions. During the outdoor sequence, cool streetlight blues contrast with the warmth of indoor lighting, emphasizing the protagonist’s solitude and introspection.",
                                "style": "The visual style is minimalist yet emotionally evocative, with a strong focus on the subtleties of character expressions and body language. The framing and composition are clean and intentional, drawing the viewer’s eye to the key moments of interaction. The use of shallow depth of field isolates the characters from their environment, enhancing the emotional weight of each scene. The lighting and color grading subtly shift to reflect the evolving mood, from warmth and camaraderie to isolation and eventual hope.",
                                "parts": {
                                    "part1": "The scene opens in a bustling restaurant filled with ambient chatter and the clinking of cutlery. The lighting is dim, with warm, golden overhead fixtures casting a soft glow on the diners. The protagonist, a person in their mid-20s with casual yet thoughtful attire, sits at a round table surrounded by friends. They lean forward slightly, engaged in the lively conversation, their smile genuine but tinged with a hint of introspection.",
                                    "part2": "One of the friends, dressed in a vibrant red sweater that stands out against the muted tones of the room, delivers a lighthearted but cutting joke. Their laughter echoes across the table, drawing smiles and chuckles from the group. The camera zooms in on the protagonist’s face as their smile falters, the subtle shift in expression revealing a mix of hurt and unease.",
                                    "part3": "The protagonist responds sharply, their tone firm but not loud, cutting through the jovial atmosphere. The laughter abruptly stops, and an uneasy silence falls over the table. The camera captures a series of close-ups: a friend nervously stirring their drink, another avoiding eye contact, and the protagonist looking down at their hands, regret flickering across their face.",
                                    "part4": "The setting shifts to a quiet street illuminated by cool, bluish streetlights. The protagonist walks alone, their hands in their pockets and their shoulders slightly hunched. The distant hum of passing cars and the rustling of leaves fill the silence. The camera lingers on their face, which reflects a mix of guilt, frustration, and self-reflection. Shadows from the streetlights play across their features, adding depth to the scene.",
                                    "part5": "In their small, dimly lit apartment, the protagonist sits on the edge of their bed, their phone glowing in their hands. They type a message, pausing occasionally to think. The screen reveals their words: 'I’m sorry for what I said. I shouldn’t have reacted that way, but I was hurt. I hope we can talk.' The camera captures their expression, a blend of vulnerability and resolve, as they hit send.",
                                    "part6": "The final scene shows the protagonist sitting by the window, the early morning light casting soft shadows across the room. Their phone buzzes, and they pick it up to read the reply: 'Let’s talk soon.' A faint smile plays on their lips, and the camera slowly pulls back, framing them against the backdrop of the city waking up. The scene ends on a note of quiet reconciliation and renewed hope."
                                    }
                                }
                            }
                            """ % story

                            formatted = chat_response(model_name, formatting_system_context, formatting_prompt, temperature=0.2, mode="formatted")
                            #print(f"Formatted response: {formatted}")
                            obj = json.loads(formatted)
                        except Exception as e:
                            logging.error(f"An error occurred during llm generation: {e}")
                            continue
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
                            continue


                        formatting_system_context = """You are now a expert filmographer and understander who must provide perfect summaries and extract the actual answer and question while being able to make detailed descriptions of answered questions in JSON."""

                        formatting_prompt = """

                        Here is the answer to a question you created:
                        %s

                        Create a JSON object with the following:

                        "title": The question that you were answering (make it captivating by adding something shocking about the question to the title),
                        "body": the answer you created
                        "hastags": a seires of single word strings to use to make it captivating i.e entertaining, satifying etc.
                        "description": a description of the question and answer
                        "prompt": an object that will contain instructions to a text to video model that describes what is going on in the question/answer, you must describe the visual art style that should be used.
                        Ensure that chracters and environment is described thoroughly with descriptions about their appearance, age, hair, eyes, clothes, surroundings, atmosphere etc. dont not shy on the detail.
                            Generate objects for the "color", "style" and an array of text for "parts", each part should be roughly 1/6 of the total story. 
                        If no story was generated, add a key "error" and set it to true

                        Here is an example of how it should be formatted, ensure that the color, style and sscenes are MUCH MUCH MUCH more descriptive:

                        {
                        "title": "How Did You Catch Your First White Whale, Captain?",
                        "body": "Ah, the first one—it’s a tale etched into my very soul. We spotted the beast at dawn, its alabaster hide gleaming under the breaking sun. My crew and I, mere men against a leviathan, rowed out in silence. Harpoon in hand, I struck true, but the whale dragged us miles before we wore it down. Blood mingled with the sea, and I knew then that I was forever bound to the hunt.",
                        "hashtags": ["whaling", "adventure", "ocean", "18thCentury", "drama"],
                        "description": "The gripping tale of a captain recalling his first encounter with a white whale, a story of determination, courage, and obsession on the high seas.",
                        "prompt": {
                            "color": "The palette captures the somber mood of the sea with dark, stormy grays and deep navy blues, creating a sense of foreboding and immensity. Flashes of vivid crimson streak through the ocean, signifying both the peril of the hunt and the visceral triumph of the crew's efforts. Subtle hints of golden sunlight and muted lavender hues appear at the edges of the frames, marking the passage of time from the cold light of dawn to the warm, fading glow of dusk. Shadows play an integral role, lending depth and texture to the scenes and enhancing the sense of drama.",
                            "style": "The visual style is painterly and deeply atmospheric, evoking the timeless aesthetic of oil paintings from the maritime golden age. Each frame feels like a masterfully rendered tableau, rich in texture and alive with intricate details that pull the viewer into the story. The chiaroscuro technique highlights the stark contrasts between light and shadow, emphasizing the interplay of man, sea, and beast. The sweeping compositions evoke grandeur and scale, while intimate close-ups capture raw human emotion. Movement is deliberate and evocative, bringing to life the tension and poetry of the epic confrontation.",
                            "parts": {
                                "part1": "The scene opens with the captain, a weathered man in his late 40s with a salt-streaked beard, gripping the ship’s wheel. His piercing eyes scan the horizon as the creaking ship cuts through turbulent waters. The ocean stretches endlessly, a vast expanse of restless gray, as the rising sun struggles to pierce the heavy mist, casting a pale, ghostly light.",
                                "part2": "The focus shifts to the crew, huddled in a rowboat bobbing precariously on the waves. Their muscles strain as they row toward a shadow beneath the surface, a pale, glistening shape moving with ominous grace. The soundscape is dominated by the rhythmic crashing of waves, the creak of wood, and the labored breaths of the men, amplifying the tension in the air.",
                                "part3": "The harpoon is hurled with precision, striking the white whale’s side. The creature reacts with a thunderous thrash, its immense tail breaking the surface in an explosion of foam and blood. Crimson streaks stain the water as the spray drenches the crew. The camera captures the captain’s stoic resolve, his unwavering gaze locked onto the majestic yet menacing beast.",
                                "part4": "The pursuit intensifies as the whale dives, pulling the rowboat dangerously close to capsizing. Waves churn violently, and the boat sways precariously. The crew’s faces, framed in close-up, reveal a mix of terror, determination, and awe. The tension builds as the line tightens and the whale resurfaces briefly, its enormous body glistening in the dim light.",
                                "part5": "The white whale finally emerges, weakened but defiant. The crew works with grim efficiency, their movements synchronized in a somber rhythm. The sea turns a deep, unsettling red as the creature’s strength ebbs. Their faces reflect a complex mix of triumph, exhaustion, and a hint of sorrow as they grapple with the weight of their victory.",
                                "part6": "The final scene lingers on the captain, standing alone on the deck of the ship. The golden hues of dusk bathe the scene in soft, melancholic light. He gazes out over the endless ocean, his expression a profound mixture of weariness, reflection, and an intangible shadow of the hunt that now haunts his soul. The faint cry of gulls and the steady lapping of waves provide a quiet, contemplative close."
                                }
                            }
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
                            Generate objects for the "color", "style", and an array of text for "parts" (each part should be roughly 1/6 of the total story). 
                        If no story was generated, add a key "error" and set it to true.

                        Here is an example of how it should be formatted, ensure that the color, style and sscenes are MUCH MUCH MUCH more descriptive:

                        {
                        "title": "The Hunt Beneath Crimson Waves",
                        "body": "The dawn broke with a somber light, casting its weak rays over the tumultuous sea. Captain Rourke stood at the prow, his piercing gaze fixed on the horizon. The white whale breached the waves, its alabaster body glistening, a fleeting specter of myth. The chase was on, harpoons soaring through the salty air, cries of men mingling with the ocean's roar. Hours later, as the creature's strength waned, Rourke stood triumphant, though the weight of the kill bore heavy on his soul.",
                        "hashtags": ["adventure", "whaling", "maritime", "18thCentury", "drama"],
                        "description": "An evocative tale of Captain Rourke's relentless pursuit of a legendary white whale, capturing the struggle between man and nature, triumph, and guilt.",
                        "prompt": {
                            "color": "A palette dominated by deep, inky ocean blues that convey the vastness and mystery of the sea, juxtaposed with stark, almost blinding whites to capture the ethereal presence of the whale and the misty horizon. Flashes of vibrant crimson punctuate the scenes, symbolizing danger, life, and the visceral reality of the hunt. The colors shift subtly with the changing light of day, from muted grays of morning mist to the fiery oranges and purples of dusk, evoking a sense of time and the relentless passage of the hunt.",
                            "style": "Cinematic and profoundly dramatic, with compositions inspired by the chiaroscuro contrasts of classic maritime paintings. The interplay of light and shadow is used to emphasize the enormity of the whale and the fragility of the human figures against the vast expanse of the sea. Every frame is imbued with a painterly quality, where rich textures and meticulous details bring the maritime world to life. The storytelling is heightened by sweeping, dynamic camera movements, evoking the grandeur of epic films, while intimate close-ups capture the raw emotion and resolve etched into the faces of the crew.",
                            "parts": {
                                "part1": "The scene begins with Captain Rourke, a seasoned mariner whose weathered face tells tales of countless storms and battles, standing on the deck of the *Resolute*. The ship creaks and groans against the rolling waves of a vast and gray ocean, shrouded in an almost otherworldly mist. The crew, clad in worn oilskins, moves with practiced efficiency, though a nervous tension lingers in the air. The distant cry of a gull echoes, barely audible over the rhythmic crash of the sea. Rourke’s steely gaze cuts through the fog as he grips the railing, his knuckles white, scanning for a shadow in the depths.",
                                "part2": "Without warning, the white whale breaches the surface in a breathtaking explosion of water and power. Its massive, ghostly form is both awe-inspiring and terrifying, glistening in the muted sunlight. The crew freezes momentarily, caught between fear and wonder, before scrambling into small rowboats. Oars splash as they push away from the ship, their movements frantic but coordinated. The tension thickens, the sound of their ragged breaths and creaking oarlocks punctuated by the whale’s deep, resonant exhale. Rourke’s voice cuts through the chaos, calm yet commanding, urging the men forward. His eyes are locked on the creature, a symbol of both destiny and obsession.",
                                "part3": "The harpoons are hurled with precision honed by years of practice, their steel tips gleaming as they arc through the air. One strikes true, embedding deep into the whale’s thick hide. A haunting, guttural bellow reverberates across the water as the creature thrashes in agony, its powerful tail churning the ocean into a frothy tempest. One of the rowboats is caught in the maelstrom, its occupants thrown into disarray. Rourke, his voice unwavering, shouts commands to the remaining boats, coordinating their assault with the precision of a battlefield general. The whale’s movements become erratic, its immense strength both a weapon and a testament to its will to survive.",
                                "part4": "Hours pass as the chase drags on, the relentless pursuit pushing the crew to their physical and mental limits. The once-calm sea grows restless, its surface dark and foreboding under a sky streaked with ominous clouds. The crew’s weariness is evident in their slumped shoulders and labored movements, yet they press on, driven by the unyielding determination of their captain. Rourke remains a pillar of focus and resolve, his weathered hands steady on the tiller as he calculates every move. The whale, though formidable, shows signs of fatigue, its breaches less forceful, its movements slower. The clash between man and nature becomes a testament to endurance and sheer will.",
                                "part5": "As dusk approaches, the whale surfaces for what seems to be the final time. Its once-majestic form is now battered and bloodied, the sea around it tinged crimson. The crew works silently, their faces a mixture of awe, sorrow, and grim determination. The harpoons are retrieved, the lines tightened, and the final blows are delivered with a reverence that belies the violence of the act. The ocean, a silent witness, reflects the deep hues of the setting sun, casting an almost ethereal glow over the somber scene. The men, though victorious, are subdued, their triumph tempered by the weight of their actions.",
                                "part6": "The story concludes with Captain Rourke standing alone on the deck of the *Resolute*, the day's events etched deeply into his weary expression. The ship drifts in the twilight, its sails catching the last whispers of the dying breeze. Rourke’s gaze is fixed on the horizon, where the sea and sky meet in a fleeting embrace of gold and violet. The shadow of the hunt lingers around him, a reminder of the cost of obsession and the fragile line between victory and loss. The crew below decks celebrates quietly, their voices muted, as Rourke reflects on the profound and enduring bond between man and the untamed forces of nature."
                            }
                        }
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

                "style" describes what kind of film to take inspiration from.
                For example "style" should look like: "A disney film from the 90s" or "A pixar film" or "A BBC telenovela".
                "color" should be a list of adjectives that describe the mood, appearance of the characters and vibe that a film adaptation of your story would have using different accents, hues tones etc.
                For example "color" should look like: "color": "A palette dominated by deep, inky ocean blues that convey the vastness and mystery of the sea, juxtaposed with stark, almost blinding whites to capture the ethereal presence of the whale and the misty horizon. Flashes of vibrant crimson punctuate the scenes, symbolizing danger, life, and the visceral reality of the hunt. The colors shift subtly with the changing light of day, from muted grays of morning mist to the fiery oranges and purples of dusk, evoking a sense of time and the relentless passage of the hunt.",

                
                Each part within "parts" should be expressed so that a text to video model can create a detailed and comphrensive scene.
                Each of the parts have no context of the previous ones, each part must fully describe the characters appearance, interactions as well as the environment and actions. 
                Each of the characters must be incredibly detailed, especially their faces height and age. The environment must be clear and descriptive.
                Follow the provided schema for JSON
                """ % json.dumps(obj)

                formatted = generate_response(model_name, formatting_system_context, formatting_prompt, temperature=0.63, mode="formatted")
                obj = json.loads(formatted)
                #endregion
               
                gen_id = str(uuid.uuid4())
                if len(obj) == 0:
                    logging.error("Unable to extrat json story")
                    continue
                write_json_to_folder(obj, "data/stories", gen_id + ".json")

                try:
                    if generate_ai_video(obj, gen_id) < 0:
                        continue
                    path = os.path.join(r"D:\utils\ComfyUI_windows_portable", "ComfyUI", "output", gen_id) 
                    videos = [file for file in os.listdir(path) if file.endswith(".mp4")]
                    for video in videos:
                        interpolate_ai_video(os.path.join(path, video))
                    return

                    full = generate_tts_for_post(obj, tts_folder_name=gen_id)

                    if full != None:
                        # Create a combined video for the post
                        if (create_combined_video_for_post(obj, full, video_clips=videos) != None):
                            logging.info(f"Succesffuly generated post: {obj['title']}")
                    else:
                        logging.error(
                            f"Failed to generate TTS for post: {obj['title']}")

                except Exception as e:
                    logging.error(f"An error occurred during video generation: {e}") 

            except Exception as e:
                print(f"An error occurred: {e}")

                


                # # title, content = generate_tts_for_post(post, tts_folder_name=tts_folder_name)
                # full = generate_tts_for_post(
                #     post, tts_folder_name=tts_folder_name)

                # # if title is not None and content is not None:



                # # Generate TTS for the post
                # # Step 3: Pause briefly before restarting the scrapers
                # logging.info(
                #     "Main thread tasks completed. Restarting scrapers in 10 seconds...")
                # sleep_for_duration(10)  # Short pause before restarting the cycle


if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(os.path.dirname(abspath))
    os.chdir(dname)


    main()


    # directory = "data/stories"  # Replace with your directory path
    # file_name = "98485939-8a8b-4ba7-825b-321a3eaa232f.json"  # Replace with your file name

    # # Construct the full file path
    # file_path = os.path.join(directory, file_name)

    # # Read the JSON file
    # try:
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         data = json.load(file)  # Load the JSON data as a Python dictionary
    #         generate_and_upscale_video(data, "98485939-8a8b-4ba7-825b-321a3eaa232f")
    # except FileNotFoundError:
    #     print(f"Error: The file at '{file_path}' was not found.")
    # except json.JSONDecodeError as e:
    #     print(f"Error decoding JSON: {e}")
