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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_random_stories_from_csv(filename='data/reddit_posts.csv', num_posts=10):
    """Pick a specified number of random stories that haven't been marked as 'generated'."""
    df = pd.read_csv(filename)
    # Filter out stories that have already been generated
    unprocessed_posts = df[(df['post_type'] != 'generated') & (df['post_type'] != 'video')]
    
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
        # Generate TTS for the title
        title_path, timings_title = get_tts(title, f"data/TTS/{tts_folder_name}", "title.mp3", speech_rate=1)
        
        # Generate TTS for the content
        content_path, timings_content = get_tts(content, f"data/TTS/{tts_folder_name}", "content.mp3", speech_rate=1)
        
        logging.info(f"TTS generation successful for post: {title}")
        return (title_path, timings_title, post['title']), (content_path, timings_content, post['content'])
    except Exception as e:
        logging.error(f"Error generating TTS for post {title}: {e}")
        return None, None

def update_post_as_generated(post_id, filename='data/reddit_posts.csv'):
    """Update the post as 'generated' in the CSV."""
    df = pd.read_csv(filename)
    df.loc[df['id'] == post_id, 'post_type'] = 'generated'
    df.to_csv(filename, index=False)
    logging.info(f"Marked post with ID {post_id} as 'generated'.")

def sleep_for_duration(seconds):
    """Function to sleep in a non-blocking manner using threading."""
    time.sleep(seconds)

def main():
    """Main function to run the scraping, TTS, and video generation cycle."""
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(os.path.dirname(abspath))
    os.chdir(dname)

    interval = 30  # Duration for scrapers to run before switching to main thread tasks

    while True:
        # Step 1: Run scrapers in parallel for the specified interval
        # run_scrapers(interval)

        # Step 2: Process posts after scrapers are done
        logging.info("Scraping completed. Now selecting and processing posts.")

        # Fetch multiple posts to process
        posts = get_random_stories_from_csv()
        
        for post in posts:
            if post is not None:
                logging.info(f"Selected post: {post['title']} (ID: {post['id']})")

                # Generate TTS for the post
                tts_folder_name = post['id']
                title, content = generate_tts_for_post(post, tts_folder_name=tts_folder_name)
                
                if title is not None and content is not None:
                    # Mark the post as generated

                    # Create a combined video for the post
                    if (create_combined_video_for_post(post, title, content) != None):
                        update_post_as_generated(post['id'])

                else:
                    logging.error(f"Failed to generate TTS for post: {post['title']}")
            else:
                logging.info("No unprocessed posts available.")

        # Step 3: Pause briefly before restarting the scrapers
        logging.info("Main thread tasks completed. Restarting scrapers in 10 seconds...")
        sleep_for_duration(10)  # Short pause before restarting the cycle

if __name__ == "__main__":
    main()
