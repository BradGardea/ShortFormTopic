import random
import logging
import time
import threading
from reddit import run_scrapers
# from eleven_labs import get_tts, get_timings
from azure_synth import get_tts
from video_generator import create_combined_video_for_post
import pandas as pd
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_random_stories_from_csv(filename='data/reddit_posts.csv', num_posts=10):
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
    content = post['content']

    try:
        # # Generate TTS for the title
        # title_path, timings_title = get_tts(title, f"data/TTS/{tts_folder_name}", "title", speech_rate=1)

        # # Generate TTS for the content
        # content_path, timings_content = get_tts(content[:200], f"data/TTS/{tts_folder_name}", "content", speech_rate=1)

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


def main(title = None, content = None):
    """Main function to run the scraping, TTS, and video generation cycle."""
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(os.path.dirname(abspath))
    os.chdir(dname)

    interval = 30  # Duration for scrapers to run before switching to main thread tasks
    if content != None:
        post = {"title": title, "id": "motivation_1", "content": content }

        logging.info(f"Selected post: {post['title']} (ID: {post['id']})")

        # Generate TTS for the post
        tts_folder_name = post['id']
        # title, content = generate_tts_for_post(post, tts_folder_name=tts_folder_name)
        full = generate_tts_for_post(
            post, tts_folder_name=tts_folder_name)

        # if title is not None and content is not None:
        if full != None:
            # Create a combined video for the post
            if (create_combined_video_for_post(post, full) != None):
                # update_post_as_generated(post['id'])
                logging.info("Exiting")
        else:
            logging.error(
                f"Failed to generate TTS for post: {post['title']}")

    else:
        while True:
            # Step 1: Run scrapers in parallel for the specified interval
            # run_scrapers(interval)

            # Step 2: Process posts after scrapers are done
            # logging.info("Scraping completed. Now selecting and processing posts.")

            # Fetch multiple posts to process
            posts = get_random_stories_from_csv()

            for post in posts:
                if post is not None:
                    logging.info(
                        f"Selected post: {post['title']} (ID: {post['id']})")

                    # Generate TTS for the post
                    tts_folder_name = post['id']
                    # title, content = generate_tts_for_post(post, tts_folder_name=tts_folder_name)
                    full = generate_tts_for_post(
                        post, tts_folder_name=tts_folder_name)

                    # if title is not None and content is not None:
                    if full != None:
                        # Create a combined video for the post
                        if (create_combined_video_for_post(post, full) != None):
                            update_post_as_generated(post['id'])
                    else:
                        logging.error(
                            f"Failed to generate TTS for post: {post['title']}")
                else:
                    logging.info("No unprocessed posts available.")

            # Step 3: Pause briefly before restarting the scrapers
            logging.info(
                "Main thread tasks completed. Restarting scrapers in 10 seconds...")
            sleep_for_duration(10)  # Short pause before restarting the cycle


if __name__ == "__main__":
#     text = '''
# There will come moments in life when the crowd disappears, the applause stops, and the people you thought would always stand by you are nowhere to be found. It’s in these moments that you face the ultimate test of strength—not just physical, but mental, emotional, and spiritual.

# When no one else believes in you, you must believe in yourself. When the world turns its back, you must step forward and face it head-on. The truth is, every person who has ever achieved something great has walked this lonely road.

# Think about the tallest trees in the forest. They didn’t grow tall because the sun shone on them every day or because the rain always came when they needed it. They grew tall because they had no choice but to dig their roots deeper when the storms came. They thrived because they faced resistance, because they refused to fall.

# You are that tree. When the storm hits, dig deep. When you feel isolated, remember this: isolation is where greatness begins. It’s where you discover your strength, where you build your character, and where you find out what you’re truly made of.

# Success isn’t about having a crowd cheering you on—it’s about showing up for yourself when no one else will. It's about waking up every morning and deciding, "I’m going to try again." Not for anyone else, but because you owe it to yourself.

# Every small step forward, no matter how insignificant it may seem, is a victory. Every moment you choose to keep going, even when it feels like the world is against you, is a testament to your resilience.

# You are stronger than you know. You’ve faced challenges before, and you’re still standing. You’ve overcome things that you thought would break you. So when you feel like giving up, remind yourself: this is just one more challenge, one more hurdle. And you will rise again, just like you always have.

# Remember, you don’t need the world to believe in you. You just need you to believe in you. Keep going. Keep growing. Keep fighting. Because the best chapters of your story are yet to be written—and they’re going to be incredible.

# You’ve got this.

# Keep pushing forward, and you'll amaze yourself with what you can achieve! 🌟'''
#     main("When No One Else is There for You, Be There for Yourself", text)
    main()
