import logging
import pandas as pd
from time import sleep
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
import csv
import os
import subprocess
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

stop_event = Event()

def scrape_subreddit_with_timeout(subreddit_url, existing_ids, subreddit_type, interval, stop_event):
    """ Wrapper to handle the timeout and stop signal for scraping a subreddit. """
    driver = webdriver.Chrome()
    driver.get(subreddit_url)
    
    # Run the scroll and extract function, passing the stop_event to check if it's time to stop
    scroll_and_extract_data(driver, subreddit=subreddit_url.split("/")[-2], interval=interval, type=subreddit_type, existing_ids=existing_ids, stop_event=stop_event)
    
    driver.quit()


def load_existing_ids(filename='./data/reddit_posts.csv'):
    """ Load existing post IDs from CSV into a set for quick lookup. """
    try:
        existing_data = pd.read_csv(filename)
        existing_ids = set(existing_data['id'])
        logging.info(f"Loaded existing IDs from {filename}. Found {len(existing_ids)} IDs.")
    except FileNotFoundError:
        existing_ids = set()
        logging.info(f"{filename} not found. No existing IDs to load.")
    return existing_ids


def scroll_and_extract_data(driver, subreddit, existing_ids, interval = 120, type="text", stop_event=None):
    """ Scroll the page and extract data based on the subreddit type, with a stop event check. """
    logging.info(f"Starting to scroll the page for new posts in {subreddit}...")
    last_height = driver.execute_script("return document.body.scrollHeight")
    start_time = time.time()

    while True:
        # Check if we have exceeded the time limit (2 minutes)
        if time.time() - start_time > interval:  # 120 seconds = 2 minutes
            logging.info(f"{interval} seconds elapsed. Stopping scraper.")
            break

        # Scroll down to the bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        logging.info("Scrolling down...")
        sleep(2)  # Wait for new content to load

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            logging.info("No new content loaded. Stopping scrolling.")
            break
        last_height = new_height

        # Extract data after scrolling
        if type == 'text':
            new_posts = extract_text_posts(driver, existing_ids)
        elif type == 'video':
            new_posts = extract_video_posts(driver, existing_ids)

        if new_posts:
            logging.info(f"Found {len(new_posts)} new posts in {subreddit}.")
            append_to_csv(new_posts)

        # Check if stop event is set (in case we are in the middle of scraping and want to stop)
        if stop_event.is_set():
            logging.info("Stop event triggered. Finishing current post...")
            break

def extract_text_posts(driver, existing_ids):
    """ Extract text posts for the r/AmItheAsshole subreddit. """
    posts = driver.find_elements(By.TAG_NAME, 'shreddit-post')
    new_posts = []

    for post in posts:
        post_id = post.get_attribute('id')
        if post_id in existing_ids:
            logging.info(f"Skipping already added post: {post_id}")
            continue

        title = post.get_attribute('post-title')
        author = post.get_attribute('author')
        timestamp = post.get_attribute('created-timestamp')
        score = post.get_attribute('score')
        comment_count = post.get_attribute('comment-count')
        permalink = post.get_attribute('content-href')

        # Wait explicitly for paragraphs to load within the current post element
        WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, 'p'))
        )

        paragraphs = post.find_elements(By.TAG_NAME, 'p')
        content = "\n".join([para.get_attribute('innerText').strip() for para in paragraphs if para.get_attribute('innerText').strip()])
        post_type = 'text'
        if "AITA" or "WIBTA" in title:
            post_type = "AITA"
        new_post = {
            'author': author,
            'title': title,
            'timestamp': timestamp,
            'score': score,
            'comment_count': comment_count,
            'permalink': permalink,
            'content': content,
            'id': post_id,
            'post_type': post_type
        }

        new_posts.append(new_post)
        logging.info(f"Extracted text post: {title} by {author} with ID: {post_id}")

    return new_posts

def download_video(url, output_dir, output_filename):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path for the output file
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        return

    # Check if the URL is a GIF
    if '.gif' in url:
        logging.info(f"Detected GIF: {url}. Converting to MP4...")
        # Command to convert GIF to MP4 using ffmpeg
        command = [
            "ffmpeg",
            "-i", url,  # Input GIF file
            "-c:v", "libx264",  # Use H.264 codec for the MP4
            "-pix_fmt", "yuv420p",  # Set pixel format to make it compatible with most players
            "-hide_banner",
            "-loglevel", "error",
            output_path
        ]
    else:
        # Command to download and convert the .m3u8 to .mp4 using ffmpeg
        command = [
            "ffmpeg",
            "-i", url,  # Input URL (m3u8 file)
            "-c", "copy",    # Copy codec without re-encoding
            "-bsf:a", "aac_adtstoasc",  # Fix audio stream
            "-hide_banner", 
            "-loglevel", "error",
            output_path
        ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        logging.info(f"Downloaded and converted to MP4: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading video: {e}")


def sanitize_title(title):
    # Convert to lowercase
    title = title.lower()
    # Remove special characters and punctuation
    title = re.sub(r'[^\w\s-]', '', title)
    # Replace spaces with underscores
    title = re.sub(r'\s+', '_', title)
    return title

def extract_video_posts(driver, existing_ids):
    """ Extract video posts for the r/oddlysatisfying subreddit with shadow DOM handling. """
    posts = driver.find_elements(By.TAG_NAME, 'shreddit-post')
    new_posts = []

    for post in posts:
        post_id = post.get_attribute('id')
        if post_id in existing_ids:
            logging.info(f"Skipping already added post: {post_id}")
            continue

        title = post.get_attribute('post-title')
        author = post.get_attribute('author')
        timestamp = post.get_attribute('created-timestamp')
        score = post.get_attribute('score')
        comment_count = post.get_attribute('comment-count')
        permalink = post.get_attribute('content-href')

        # Accessing shadow DOM to get video URL
        try:
            aspect_ratio = post.find_element(By.TAG_NAME, "shreddit-aspect-ratio")
            if not aspect_ratio:
                logging.info(f"No video found for post ID: {post_id}")
                continue
            async_loader = aspect_ratio.find_element(By.TAG_NAME, "shreddit-async-loader")
            if not async_loader:
                logging.info(f"No video found for post ID: {post_id}")
                continue
            shreddit_player = async_loader.find_element(By.TAG_NAME, "shreddit-player-2")
            if not shreddit_player:
                logging.info(f"No video found for post ID: {post_id}")
                continue
            video_url = None

            if (shreddit_player):
                video_url = shreddit_player.get_attribute('src')

            if not video_url:
                logging.info(f"No video found for post ID: {post_id}")
                continue
            else:
                if ".gif" in video_url:
                    download_video(video_url, ".\\data\\satisfying_videos\\loops", sanitize_title(title) + ".mp4" )
                else:
                    download_video(video_url, ".\\data\\satisfying_videos\\regular", sanitize_title(title) + ".mp4" )


            new_post = {
                'author': author,
                'title': title,
                'timestamp': timestamp,
                'score': score,
                'comment_count': comment_count,
                'permalink': permalink,
                'video_url': video_url,
                'id': post_id,
                'post_type': 'video'
            }

            new_posts.append(new_post)
            logging.info(f"Extracted video post: {title} by {author} with ID: {post_id}")

        except Exception as e:
            logging.error(f"Error getting data for post {post_id}: {e}")
            continue

    return new_posts

def append_to_csv(new_posts, filename='.\\data\\reddit_posts.csv'):
    new_data = pd.DataFrame(new_posts)

    try:
        existing_data = pd.read_csv(filename)
        logging.info(f"Loaded existing data from {filename}.")
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=['author', 'title', 'timestamp', 'score', 'comment_count', 'permalink', 'content', 'video_url', 'id', 'post_type'])
        logging.info(f"{filename} not found. Creating a new file.")

    # Concatenate existing data with new posts
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Remove duplicates
    combined_data.drop_duplicates(subset=['title', 'author', 'timestamp'], inplace=True)

    # Save to CSV
    combined_data.to_csv(filename, index=False, quotechar='"', quoting=csv.QUOTE_ALL)
    logging.info(f"Appended {len(new_posts)} new posts to {filename}.")


def scrape_subreddit(subreddit_url, existing_ids, subreddit_type):
    driver = webdriver.Chrome()
    driver.get(subreddit_url)
    scroll_and_extract_data(driver, subreddit=subreddit_url.split("/")[-2], type=subreddit_type, existing_ids=existing_ids)
    driver.quit()


def run_scrapers(interval=30):
    """Runs multiple subreddit scrapers in parallel for a specified interval, then stops them."""
    while True:
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Load existing IDs once and pass them to the threads
            existing_ids = load_existing_ids()

            # Start the scraper tasks
            stop_event.clear()  # Ensure stop event is clear at the start
            future1 = executor.submit(scrape_subreddit_with_timeout, "https://www.reddit.com/r/AmItheAsshole/", existing_ids, "text", interval, stop_event)
            # future2 = executor.submit(scrape_subreddit_with_timeout, "https://www.reddit.com/r/perfectloops/", existing_ids, "video", interval, stop_event)
            # future3 = executor.submit(scrape_subreddit_with_timeout, "https://www.reddit.com/r/oddlysatisfying/", existing_ids, "video", interval, stop_event)
            # future4 = executor.submit(scrape_subreddit_with_timeout, "https://www.reddit.com/r/Satisfyingasfuck/", existing_ids, "video", interval, stop_event)
            # future5 = executor.submit(scrape_subreddit_with_timeout, "https://www.reddit.com/r/mildlyinfuriating/", existing_ids, "text", interval, stop_event)


            # Wait for the specified interval
            logging.info(f"Scraping for {interval} seconds...")
            sleep(interval)
            
            # Signal the scrapers to stop gracefully
            stop_event.set()
            logging.info("Stopping scrapers...")

            # Wait for all scraper threads to complete
            wait([future1])
            logging.info("All scrapers have finished. Returning control to the main thread.")

        # Break out of the while loop to return control to the main thread
        break
