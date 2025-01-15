import requests
import random
from bs4 import BeautifulSoup
import csv
import os
import subprocess
import shutil
import json
import re
import time
import ctypes
from threading import Thread
from faster_whisper import WhisperModel
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Access Twitch credentials from the environment
client_id = os.getenv('TWITCH_CLIENT_ID')
access_token = os.getenv('TWITCH_ACCESS_TOKEN')

# Check if credentials are provided
if not client_id or not access_token:
    raise EnvironmentError(
        "Twitch credentials are missing. Please set 'TWITCH_CLIENT_ID' and 'TWITCH_ACCESS_TOKEN' in a .env file."
    )

# Define the base path for the transcripts folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
BASE_TRANSCRIPTS_FOLDER = os.path.join(SCRIPT_DIR, "transcripts")  # Folder named "transcripts" in the script's directory
VTUBERS_CSV = os.path.join(BASE_TRANSCRIPTS_FOLDER, "verified_vtubers.csv")
VODS_CSV = os.path.join(BASE_TRANSCRIPTS_FOLDER, "valid_vods.csv")

# Function to clean and normalize Twitch URLs
def clean_url(url):
    if url:
        url = url.replace("m.twitch.tv", "twitch.tv")
        url = url.replace("?desktop-redirect=true", "")
        if url.endswith("/"):
            url = url[:-1]  # Remove the trailing slash if it exists (which it will, god hates me)
    return url

# Function to get pages from a category on Fandom
def get_category_pages(category_url):
    pages = []
    next_page = category_url

    while next_page:
        try:
            print(f"Fetching category page: {next_page}")
            response = requests.get(next_page)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.select(".category-page__member-link"):
                page_url = f"https://virtualyoutuber.fandom.com{link['href']}"
                if not any(page_url.split("/")[-1].startswith(prefix) for prefix in ["User:", "Draft:"]):
                    pages.append(page_url)

            next_link = soup.select_one(".category-page__pagination-next")
            if next_link:
                next_page = next_link["href"]
                if not next_page.startswith("http"):
                    next_page = f"https://virtualyoutuber.fandom.com{next_page}"
            else:
                break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching pages: {e}")
            break

    print(f"Collected {len(pages)} pages from category: {category_url}")
    return pages

# Function to get the intersection of VTubers from both categories
def get_verified_vtubers(twitch_category_pages, english_category_pages):
    verified_vtubers = list(set(twitch_category_pages) & set(english_category_pages))
    print(f"Found {len(verified_vtubers)} VTubers present in both categories.")
    return verified_vtubers

# Function to extract Twitch link from a VTuber page
def extract_twitch_link(page_url):
    try:
        print(f"Extracting Twitch link from: {page_url}")
        response = requests.get(page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        twitch_link = None
        for stream_link in soup.find_all("a", href=True):
            if "twitch.tv" in stream_link["href"]:
                twitch_link = clean_url(stream_link["href"])
                break

        vtuber_name = page_url.split("/")[-1].replace("_", " ")
        print(f"Extracted Twitch link for {vtuber_name}: {twitch_link}")
        return twitch_link
    except requests.exceptions.RequestException as e:
        print(f"Error extracting Twitch link from {page_url}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {page_url}: {e}")

    return None

# Function to download the VTuber's Fandom page once per VTuber
def download_vtuber_wiki_page(username, vtuber_page_url):
    vtuber_folder = os.path.join(BASE_TRANSCRIPTS_FOLDER, username)
    os.makedirs(vtuber_folder, exist_ok=True)
    wiki_page_path = os.path.join(vtuber_folder, f"{username}_wiki_page.html")

    if os.path.exists(wiki_page_path):
        print(f"Wiki page for {username} already exists. Skipping download.")
        return

    try:
        print(f"Downloading Fandom wiki page for {username}...")
        response = requests.get(vtuber_page_url)
        response.raise_for_status()
        with open(wiki_page_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"Wiki page saved to: {wiki_page_path}")
    except Exception as e:
        print(f"Failed to download Fandom page for {username}: {e}")

def load_dll(dll_path):
    """Helper function to load a DLL."""
    try:
        print(f"Loading DLL: {dll_path}")
        ctypes.WinDLL(dll_path)
        print(f"Successfully loaded DLL: {dll_path}")
    except OSError as e:
        print(f"Failed to load DLL: {dll_path}")
        raise RuntimeError(f"Failed to load {dll_path}: {e}")


def load_dll(dll_path):
    """Helper function to load a DLL."""
    try:
        print(f"Loading DLL: {dll_path}")
        ctypes.WinDLL(dll_path)
        print(f"Successfully loaded DLL: {dll_path}")
    except OSError as e:
        print(f"Failed to load DLL: {dll_path}")
        raise RuntimeError(f"Failed to load {dll_path}: {e}")


def preload_cudnn_dlls(cuda_bin_path):
    """Preload all required cuDNN DLLs."""
    print(f"Preloading cuDNN DLLs from: {cuda_bin_path}")
    required_dlls = [
        "cudnn_ops64_9.dll",
        "cudnn_ops_infer64_9.dll",
        "cudnn_cnn64_9.dll",
        "cublas64_12.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_heuristic64_9.dll",
    ]

    for dll in required_dlls:
        dll_path = os.path.join(cuda_bin_path, dll)
        if not os.path.exists(dll_path):
            print(f"Error: {dll} not found in {cuda_bin_path}.")
            raise FileNotFoundError(f"{dll} not found in {cuda_bin_path}. Please verify the installation.")
        print(f"Found DLL: {dll_path}")
        load_dll(dll_path)


def transcribe(audio_file, device="cpu", output_dir=None):
    """
    Transcribes an audio file using the faster-whisper library.

    Args:
        audio_file (str): Path to the audio file to transcribe.
        device (str): Device to use for transcription, e.g., "cpu" or "cuda".
        output_dir (str): Directory to save transcription files. If None, defaults to the audio file's directory.

    Returns:
        tuple: Paths to the generated transcription files (formatted, raw).
    """
    cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    model_size = "distil-large-v3"
    compute_type = "float32"
    silence_threshold = 1.5  # 1.5 seconds of silence

    print(f"Starting transcription for: {audio_file} on device: {device}")
    if device.lower() == "cuda":
        print("CUDA device detected. Preloading cuDNN DLLs...")
        preload_cudnn_dlls(cuda_bin_path)

    print(f"Initializing WhisperModel with size: {model_size}, compute_type: {compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(audio_file)
    os.makedirs(output_dir, exist_ok=True)

    file_basename = os.path.splitext(os.path.basename(audio_file))[0]
    formatted_output_file = os.path.join(output_dir, f"formatted_{file_basename}_transcription.txt")
    raw_output_file = os.path.join(output_dir, f"{file_basename}_raw_transcript.txt")

    print(f"Formatted output file will be saved as: {formatted_output_file}")
    print(f"Raw output file will be saved as: {raw_output_file}")

    # Start transcription
    start_time = time.time()
    print("Beginning transcription...")
    segments, info = model.transcribe(audio_file, beam_size=5, language="en")
    segments = list(segments)
    print("Transcription completed.")

    # Save formatted transcription to file
    print(f"Writing formatted transcription to: {formatted_output_file}")
    try:
        with open(formatted_output_file, "w", encoding="utf-8") as file:
            file.write(f"Assumed language: English\n\n")
            prev_end_time = 0.0
            buffer = ""

            for idx, segment in enumerate(segments, start=1):
                start_time_segment = segment.start
                end_time_segment = segment.end
                text = segment.text.strip()

                # Insert a blank line if silence duration is greater than the threshold
                if start_time_segment - prev_end_time > silence_threshold:
                    if buffer:
                        file.write(buffer.strip() + "\n\n")
                    buffer = ""

                buffer += f"{text} "
                prev_end_time = end_time_segment

            # Write any remaining buffer to the file
            if buffer:
                file.write(buffer.strip() + "\n\n")
        print(f"Formatted transcription successfully written to: {formatted_output_file}")
    except Exception as e:
        print(f"Error while writing formatted transcription: {e}")
        raise

    # Save raw transcription to file
    print(f"Writing raw transcription to: {raw_output_file}")
    try:
        with open(raw_output_file, "w", encoding="utf-8") as raw_file:
            prev_end_time = 0.0
            for segment in segments:
                start_time_segment = segment.start
                text = segment.text.strip()

                # Insert a new line if silence duration is greater than the threshold
                if start_time_segment - prev_end_time > silence_threshold:
                    raw_file.write("\n")

                raw_file.write(f"{text} ")
                prev_end_time = segment.end
        print(f"Raw transcription successfully written to: {raw_output_file}")
    except Exception as e:
        print(f"Error while writing raw transcription: {e}")
        raise

    total_elapsed_time = time.time() - start_time
    audio_duration_seconds = info.duration
    total_words = sum(len(segment.text.split()) for segment in segments)
    transcription_seconds_per_audio_second = total_elapsed_time / audio_duration_seconds

    print("\nTranscription metrics:")
    print(f"Total time taken: {total_elapsed_time:.2f} seconds")
    print(f"Audio duration: {audio_duration_seconds:.2f} seconds")
    print(f"Total words transcribed: {total_words}")
    print(f"Transcription seconds per audio second: {transcription_seconds_per_audio_second:.2f}")
    print(f"Returning file paths: {formatted_output_file}, {raw_output_file}")

    return formatted_output_file, raw_output_file

# Function to download Twitch VOD and chat
def download_twitch_vod_and_chat(vod, delete_mp3_after_processing=False):
    vod_url = vod['url']
    vod_id = vod_url.split("/videos/")[1]
    title = vod['title']
    username_folder = vod['channel_name']
    vod_folder = os.path.join(BASE_TRANSCRIPTS_FOLDER, username_folder, vod_id)

    print(f"Processing {title}: {vod_url}")

    os.makedirs(vod_folder, exist_ok=True)

    vod_filename = f"{vod_id}.mp4"
    chat_json_filename = f"{vod_id}_chat.json"
    chat_csv_filename = f"{vod_id}_chat.csv"
    mp3_filename = f"{vod_id}.mp3"
    info_filename = os.path.join(vod_folder, f"{vod_id}_info.txt")

    if os.path.exists(info_filename):
        print(f"Info file for VOD {title} already exists. Skipping this VOD.")
        return False

    try:
        print(f"Downloading VOD for {title}...")
        subprocess.run([
            "TwitchDownloaderCLI.exe", "videodownload",
            "--id", vod_id,
            "-o", vod_filename
        ], check=True)

        print(f"Downloading chat for {title}...")
        subprocess.run([
            "TwitchDownloaderCLI.exe", "chatdownload",
            "--id", vod_id,
            "-o", chat_json_filename
        ], check=True)

        print(f"Converting VOD to MP3 for {title}...")
        subprocess.run([
            "ffmpeg", "-i", vod_filename, "-q:a", "0", "-map", "a", mp3_filename
        ], check=True)
        os.remove(vod_filename)

        shutil.move(chat_json_filename, os.path.join(vod_folder, chat_json_filename))
        print(f"Moved chat JSON file to: {vod_folder}")

        with open(os.path.join(vod_folder, chat_json_filename), "r", encoding="utf-8") as json_file:
            chat_data = json.load(json_file)

        with open(chat_csv_filename, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Timestamp", "Username", "Message"])
            chat_messages = chat_data.get("comments", [])
            for message in chat_messages:
                timestamp = message.get("content_offset_seconds", "N/A")
                username = message.get("commenter", {}).get("display_name", "N/A")
                chat_message = message.get("message", {}).get("body", "N/A")
                csv_writer.writerow([timestamp, username, chat_message])

        shutil.move(chat_csv_filename, os.path.join(vod_folder, chat_csv_filename))
        print(f"Moved chat CSV file to: {vod_folder}")

        mp3_dest_path = os.path.join(vod_folder, mp3_filename)
        shutil.move(mp3_filename, mp3_dest_path)
        print(f"Moved MP3 file to: {mp3_dest_path}")

        transcription_thread = run_transcription_in_thread(mp3_dest_path, vod_folder)
        print(f"Processing for {title} is complete. Transcription is running in the background.")
        return transcription_thread
    except Exception as e:
        print(f"An error occurred while processing VOD {title}: {e}")
        return False


def calculate_vod(mp4_duration):
    """
    Calculate the estimated processing time for a single MP4 file (VOD), including downloading.

    Args:
    mp4_duration (int): The duration of the MP4 file in seconds.

    Returns:
    float: The total processing time for the given MP4 in seconds.
    """
    # Example provided: 4 hours, 46 minutes, and 11 seconds MP4 file took 7 minutes and 20 seconds to download
    mp4_example_duration = 4 * 3600 + 46 * 60 + 11  # Convert 4 hours, 46 minutes, and 11 seconds to seconds
    download_time_example = 7 * 60 + 20  # Convert 7 minutes and 20 seconds to seconds

    # Calculate download ratio (download time per second of MP4)
    download_ratio = download_time_example / mp4_example_duration

    # Convert mp4 to mp3: 1/65.8th of the mp4 duration
    mp3_duration = mp4_duration / 65.8

    # Transcribe mp3: mp3_duration * 0.07913
    transcription_time = mp3_duration * 0.07913

    # Add download time (download_ratio is the time in seconds to download per second of video)
    download_time = mp4_duration * download_ratio

    # Total time for the current mp4 file (processing + transcription + downloading)
    total_time = mp3_duration + transcription_time + download_time

    # Return the total time in seconds
    return total_time * 4.5


def calculate_vods(mp4_durations):
    """
    Calculate the total processing time for multiple MP4 files (VODs) in seconds.

    Args:
    mp4_durations (list of int): The list of MP4 file durations in seconds.

    Returns:
    float: The total processing time for all MP4 files in seconds.
    """
    total_time = 0
    for mp4_duration in mp4_durations:
        total_time += calculate_vod(mp4_duration)
    return total_time  # Return raw seconds



def format_duration(seconds):
    """
    Format the given number of seconds into a human-readable string.

    Args:
    seconds (float): The total time in seconds to be formatted.

    Returns:
    str: The formatted time string.
    """
    # Extract the full seconds, minutes, hours, etc.
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    months, days = divmod(days, 30)  # Approximation
    years, months = divmod(months, 12)

    # Build the formatted string with the appropriate units
    time_str = []
    if years:
        time_str.append(f"{years} year{'s' if years > 1 else ''}")
    if months:
        time_str.append(f"{months} month{'s' if months > 1 else ''}")
    if days:
        time_str.append(f"{days} day{'s' if days > 1 else ''}")
    if hours:
        time_str.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes:
        time_str.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds:
        time_str.append(f"{seconds} second{'s' if seconds > 1 else ''}")
    if milliseconds:
        time_str.append(f"{milliseconds} millisecond{'s' if milliseconds > 1 else ''}")

    # Handle punctuation based on the number of time units
    if len(time_str) == 2:
        return f"{time_str[0]} and {time_str[1]}"
    elif len(time_str) > 2:
        return ', '.join(time_str[:-1]) + ', and ' + time_str[-1]
    else:
        return time_str[0] if time_str else '0 seconds'



# Function to get Twitch user ID
def get_user_id(channel_name, client_id, access_token):
    try:
        print(f"Fetching user ID for channel: {channel_name}")
        channel_name = channel_name.rstrip("/")
        url = f'https://api.twitch.tv/helix/users'
        headers = {
            'Client-ID': client_id,
            'Authorization': f'Bearer {access_token}'
        }
        params = {'login': channel_name}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if data['data']:
            user_id = data['data'][0]['id']
            print(f"Found user ID for {channel_name}: {user_id}")
            return user_id
    except requests.exceptions.RequestException as e:
        print(f"Error fetching user ID for {channel_name}: {e}")

    raise ValueError(f"Channel '{channel_name}' not found")

# Function to apply blacklist
def filter_blacklist(vtuber_links, blacklist):
    filtered_links = []
    for link in vtuber_links:
        channel_name = link.split("/")[-1]
        if channel_name not in blacklist:
            print(f"Channel {channel_name} passed the blacklist check!")
            filtered_links.append(link)
    return filtered_links

# Function to get recent VODs
def get_recent_vods(user_id, client_id, access_token):
    try:
        print(f"Fetching recent VODs for user ID: {user_id}")
        url = f'https://api.twitch.tv/helix/videos'
        headers = {
            'Client-ID': client_id,
            'Authorization': f'Bearer {access_token}'
        }
        params = {
            'user_id': user_id,
            'sort': 'time',
            'type': 'archive',
            'first': 100  # Fetch as many VODs as allowed by Twitch API (most vtubers don't keep more than this available, so pagination isn't worth the effort)
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        vod_data = response.json().get('data', [])

        print(f"Found {len(vod_data)} VODs for user ID: {user_id}")
        extracted_data = []
        for vod in vod_data:
            vod_url = vod.get('url')
            view_count = vod.get('view_count', 0)
            title = vod.get('title', 'No Title')
            channel_name = vod.get('user_name', 'Unknown Channel')
            duration = vod.get('duration', 'Unknown Duration')
            created_at = vod.get('created_at', 'Unknown Date')

            duration_seconds = parse_duration(duration)
            readable_duration = format_duration(duration_seconds)

            extracted_data.append({
                'url': vod_url,
                'view_count': view_count,
                'title': title,
                'channel_name': channel_name,
                'duration': readable_duration,
                'duration_seconds': duration_seconds,
                'broadcaster_id': user_id,
                'created_at': created_at
            })

        return extracted_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching VODs for user ID {user_id}: {e}")

    return []

# Function to get follower count for a streamer
def get_follower_count(broadcaster_id, client_id, access_token):
    try:
        print(f"Fetching follower count for broadcaster ID: {broadcaster_id}")
        url = f'https://api.twitch.tv/helix/channels/followers'
        headers = {
            'Client-ID': client_id,
            'Authorization': f'Bearer {access_token}'
        }
        params = {'broadcaster_id': broadcaster_id}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        follower_count = data.get('total', 0)
        print(f"Follower count for broadcaster ID {broadcaster_id}: {follower_count}")
        return follower_count
    except requests.exceptions.RequestException as e:
        print(f"Error fetching follower count for broadcaster ID {broadcaster_id}: {e}")
        return 0

# Function to parse Twitch duration (e.g., "2h30m45s") into seconds
def parse_duration(duration):
    match = re.match(r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?', duration)
    if not match:
        return 0
    hours, minutes, seconds = match.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0
    return hours * 3600 + minutes * 60 + seconds

# Function to check if a VOD is valid
def is_valid(vod, client_id, access_token):
    print(f"Checking validity of {vod['channel_name']}'s {vod['title']}")

    duration_valid = 2 * 3600 <= vod['duration_seconds'] <= 7 * 3600
    print(f"Vod duration: {vod['duration_seconds']}, is valid? - {duration_valid}")

    views_valid = vod['view_count'] >= 1000
    print(f"Vod views: {vod['view_count']}, is valid? - {views_valid}")

    follower_count = get_follower_count(vod['broadcaster_id'], client_id, access_token)
    followers_valid = follower_count >= 10000
    print(f"Channel followers: {follower_count}, is valid? - {followers_valid}")

    is_valid_vod = duration_valid and views_valid and followers_valid
    print(f"Is VOD valid? - {is_valid_vod}")
    return is_valid_vod


# Function to save verified VTubers to CSV
def save_verified_vtubers(vtubers):
    with open(VTUBERS_CSV, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["VTuber Page"])
        for vtuber in vtubers:
            writer.writerow([vtuber])
    print(f"Saved {len(vtubers)} verified VTubers to {VTUBERS_CSV}")


# Function to load verified VTubers from CSV
def load_verified_vtubers():
    if os.path.exists(VTUBERS_CSV):
        with open(VTUBERS_CSV, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            vtubers = [row[0] for row in reader]
        print(f"Loaded {len(vtubers)} verified VTubers from {VTUBERS_CSV}")
        return vtubers
    return []


# Function to save VODs to CSV
def save_vods(vods):
    with open(VODS_CSV, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=vods[0].keys())
        writer.writeheader()
        writer.writerows(vods)
    print(f"Saved {len(vods)} valid VODs to {VODS_CSV}")


# Function to load VODs from CSV
def load_vods():
    if os.path.exists(VODS_CSV):
        with open(VODS_CSV, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            vods = []
            for row in reader:
                row['duration_seconds'] = int(row['duration_seconds'])  # Convert duration_seconds to int
                vods.append(row)
        print(f"Loaded {len(vods)} valid VODs from {VODS_CSV}")
        return vods
    return []

def download_twitch_vod_and_chat(vod, delete_mp3_after_processing=False):
    vod_url = vod['url']
    vod_id = vod_url.split("/videos/")[1]
    title = vod['title']
    username_folder = vod['channel_name']
    vod_folder = os.path.join(BASE_TRANSCRIPTS_FOLDER, username_folder, vod_id)

    print(f"Processing {title}: {vod_url}")

    # Ensure folder structure exists
    print(f"Ensuring folder structure exists: {vod_folder}")
    os.makedirs(vod_folder, exist_ok=True)

    # File paths
    vod_filename = f"{vod_id}.mp4"
    chat_json_filename = f"{vod_id}_chat.json"
    chat_csv_filename = f"{vod_id}_chat.csv"
    mp3_filename = f"{vod_id}.mp3"
    info_filename = os.path.join(vod_folder, f"{vod_id}_info.txt")

    # Check if transcription files already exist
    if os.path.exists(info_filename):
        print(f"Info file for VOD {title} already exists. Skipping this VOD.")
        return False

    try:
        # Download the VOD
        print(f"Downloading VOD for {title}...")
        subprocess.run([
            "TwitchDownloaderCLI.exe", "videodownload",
            "--id", vod_id,
            "-o", vod_filename
        ], check=True)

        # Download the chat
        print(f"Downloading chat for {title}...")
        subprocess.run([
            "TwitchDownloaderCLI.exe", "chatdownload",
            "--id", vod_id,
            "-o", chat_json_filename
        ], check=True)

        # Convert VOD to MP3
        print(f"Converting VOD to MP3 for {title}...")
        subprocess.run([
            "ffmpeg", "-i", vod_filename, "-q:a", "0", "-map", "a", mp3_filename
        ], check=True)
        os.remove(vod_filename)  # Delete MP4 after conversion

        # Move chat JSON file to VOD folder
        shutil.move(chat_json_filename, os.path.join(vod_folder, chat_json_filename))
        print(f"Moved chat JSON file to: {vod_folder}")

        # Convert chat JSON to CSV
        print(f"Converting chat JSON to CSV for {title}...")
        with open(os.path.join(vod_folder, chat_json_filename), "r", encoding="utf-8") as json_file:
            chat_data = json.load(json_file)

        with open(chat_csv_filename, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Timestamp", "Username", "Message"])
            chat_messages = chat_data.get("comments", [])
            for message in chat_messages:
                timestamp = message.get("content_offset_seconds", "N/A")
                username = message.get("commenter", {}).get("display_name", "N/A")
                chat_message = message.get("message", {}).get("body", "N/A")
                csv_writer.writerow([timestamp, username, chat_message])

        shutil.move(chat_csv_filename, os.path.join(vod_folder, chat_csv_filename))
        print(f"Moved chat CSV file to: {vod_folder}")

        # Move MP3 file to VOD folder
        mp3_dest_path = os.path.join(vod_folder, mp3_filename)
        shutil.move(mp3_filename, mp3_dest_path)
        print(f"Moved MP3 file to: {mp3_dest_path}")

        # Transcribe the MP3 file from its new location
        print(f"Transcribing MP3 for {title} from {mp3_dest_path}...")
        transcription_formatted, transcription_raw = transcribe(mp3_dest_path, "cuda", output_dir=vod_folder)

        print(f"Successfully processed VOD: {title}")
        return True

    except Exception as e:
        print(f"An error occurred while processing VOD {title}: {e}")
        return False


def move_transcription_files(source_dir, filenames, destination_dir):
    """
    Moves specified files from a source directory to a destination directory, with detailed logging.

    Args:
        source_dir (str): Directory to search for the files.
        filenames (list): List of filenames to move.
        destination_dir (str): Destination directory for the files.

    Returns:
        dict: A dictionary with filenames as keys and boolean values indicating success.
    """
    print("\n--- Starting File Move Operation ---")
    print(f"Source Directory: {source_dir}")
    print(f"Destination Directory: {destination_dir}")
    print(f"Files to Move: {', '.join(filenames)}")

    results = {}
    os.makedirs(destination_dir, exist_ok=True)  # Ensure destination directory exists

    for filename in filenames:
        print(f"\nProcessing File: {filename}")
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(destination_dir, filename)

        # Log full paths
        print(f"Source Path: {source_path}")
        print(f"Destination Path: {dest_path}")

        # Check if the file exists at the source
        if os.path.exists(source_path):
            print(f"File found: {source_path}")
            try:
                # Attempt to move the file
                shutil.move(source_path, dest_path)
                print(f"Successfully moved file: {source_path} -> {dest_path}")
                results[filename] = True
            except Exception as e:
                # Log specific error during move operation
                print(f"Error moving file {source_path}: {e}")
                results[filename] = False
        else:
            # Log missing file
            print(f"File not found: {source_path}")
            results[filename] = False

    # Summary of results
    print("\n--- File Move Operation Summary ---")
    for filename, success in results.items():
        if success:
            print(f"File successfully moved: {filename}")
        else:
            print(f"File failed to move: {filename}")

    print("--- File Move Operation Complete ---\n")
    return results


# Main script
if __name__ == "__main__":
    transcription_threads = []

    print("Fetching VTuber pages from Fandom categories...")
    twitch_category_pages = get_category_pages("https://virtualyoutuber.fandom.com/wiki/Category:Twitch")
    english_category_pages = get_category_pages("https://virtualyoutuber.fandom.com/wiki/Category:English")

    print("Loading verified VTubers from CSV if it exists...")
    cached_vtubers = load_verified_vtubers()
    verified_vtuber_pages = get_verified_vtubers(twitch_category_pages, english_category_pages)

    if set(cached_vtubers) == set(verified_vtuber_pages):
        print("Cached VTuber list matches. Loading valid VODs from CSV...")
        all_vods = load_vods()
    else:
        print("Cached VTuber list does not match or is missing. Fetching new VODs...")
        save_verified_vtubers(verified_vtuber_pages)

        verified_vtuber_pages.sort(key=lambda url: url.split("/")[-1].lower())

        twitch_links = []
        vtuber_to_page = {}
        for vtuber_page in verified_vtuber_pages:
            twitch_link = extract_twitch_link(vtuber_page)
            if twitch_link:
                twitch_links.append(twitch_link)
                vtuber_to_page[twitch_link.split("/")[-1]] = vtuber_page

        blacklist = {
            "hika",
            "Rubius",
            "mikupinku",
            "yadidoll",
            "lupomarcio",
            "Vinesauce",
            "DougDoug",
            "Tectone"
        }
        twitch_links = filter_blacklist(twitch_links, blacklist)
        all_vods = []

        for link in twitch_links:
            try:
                channel_name = link.split("/")[-1]
                user_id = get_user_id(channel_name, client_id, access_token)
                vods = get_recent_vods(user_id, client_id, access_token)
                valid_vods = [vod for vod in vods if is_valid(vod, client_id, access_token)]
                if valid_vods:
                    all_vods.extend(valid_vods)
                    download_vtuber_wiki_page(channel_name, vtuber_to_page[channel_name])
            except Exception as e:
                print(f"Failed to fetch VODs for {link}: {e}")

        save_vods(all_vods)

    print(f"Total valid VODs to process: {len(all_vods)}")
    total_duration_seconds = sum(vod['duration_seconds'] for vod in all_vods)
    formatted_total_duration = format_duration(total_duration_seconds)
    print(f"Collected {len(all_vods)} valid VODs, Totaling {formatted_total_duration} of audio.")

    vod_durations = [vod['duration_seconds'] for vod in all_vods]
    estimated_time_for_all_vods_seconds = calculate_vods(vod_durations)
    estimated_time_for_all_vods = format_duration(float(estimated_time_for_all_vods_seconds))
    print(f"\nEstimated total time to process all VODs: {estimated_time_for_all_vods}")

    print(f"Starting processing for the first 3 VODs.")

    # Process the first three VODs explicitly
    if len(all_vods) < 3:
        print("Not enough VODs to process (less than 3). Exiting.")
    else:
        vod1, vod2, vod3 = all_vods[:3]

        for idx, vod in enumerate([vod1, vod2, vod3], start=1):
            print(f"\nProcessing VOD {idx}: {vod['title']} ({vod['url']})")
            try:
                success = download_twitch_vod_and_chat(vod, delete_mp3_after_processing=False)
                if success:
                    print(f"Successfully processed VOD {idx}: {vod['title']}")
                else:
                    print(f"Failed to process VOD {idx}: {vod['title']}")
            except Exception as e:
                print(f"Unexpected error during VOD {idx} processing: {e}")

    print("All selected VODs have been processed and transcribed.")
