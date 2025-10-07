import os
import json
import requests
from bs4 import BeautifulSoup
import wikipedia
import whisper
import yt_dlp as youtube_dl

# ---------------------------
# CONFIGURATION
# ---------------------------
OUTPUT_FILE = "equinet_dataset.jsonl"
DATA_DIR = "equinet_data"
LANGUAGES = ["en", "es", "fr"]  # Example languages for Wikipedia dumps
BLOG_URLS = [
]
PODCAST_URLS = [
    "https://www.youtube.com/watch?v=YooCa3A9c-0&pp=ygUZaW5kaWdlbm91cyBjbGltYXRlIGFjdGlvbg%3D%3D",
    "https://www.youtube.com/watch?v=fOMJqshXsoY&pp=ygUZaW5kaWdlbm91cyBjbGltYXRlIGFjdGlvbg%3D%3D",
    "https://www.youtube.com/watch?v=YKK7KNiAD2k&pp=ygUZaW5kaWdlbm91cyBjbGltYXRlIGFjdGlvbg%3D%3D",
    "https://www.youtube.com/watch?v=MB3Cu18Fml0&pp=ygUZaW5kaWdlbm91cyBjbGltYXRlIGFjdGlvbg%3D%3D",
]

os.makedirs(DATA_DIR, exist_ok=True)

dataset = []

# ---------------------------
# FUNCTION: Scrape Blog / Forum Content
# ---------------------------
def scrape_blog(url):
    print(f"Scraping: {url}")
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")

        posts = soup.find_all("article")
        for post in posts:
            title = post.find("h2").text if post.find("h2") else "No Title"
            content = post.get_text(separator=" ", strip=True)
            dataset.append({
                "id": f"blog_{len(dataset)+1}",
                "text": content,
                "source": url,
                "domain": "Community Blog",
                "language": "en",
                "type": "text"
            })
    except Exception as e:
        print(f"Error scraping {url}: {e}")

# ---------------------------
# FUNCTION: Download & Transcribe Podcast
# ---------------------------
def transcribe_podcast(url, index):
    print(f"Downloading and transcribing podcast: {url}")
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(DATA_DIR, f"podcast_{index}.%(ext)s"),
            "quiet": True
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_file = None
        for file in os.listdir(DATA_DIR):
            if file.startswith(f"podcast_{index}") and file.endswith((".webm", ".m4a")):
                audio_file = os.path.join(DATA_DIR, file)
                break

        if not audio_file:
            print("No podcast file found.")
            return

        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        transcript = result["text"]

        dataset.append({
            "id": f"podcast_{len(dataset)+1}",
            "text": transcript,
            "source": url,
            "domain": "Podcast",
            "language": "en",
            "type": "audio_transcript"
        })
    except Exception as e:
        print(f"Error processing podcast {url}: {e}")

# ---------------------------
# FUNCTION: Get Wikipedia Dump
# ---------------------------
def get_wikipedia(language):
    print(f"Downloading Wikipedia summaries in {language}")
    try:
        wikipedia.set_lang(language)
        for topic in ["Indigenous rights", "Climate justice", "Social equity"]:
            summary = wikipedia.summary(topic, sentences=5)
            dataset.append({
                "id": f"wiki_{len(dataset)+1}",
                "text": summary,
                "source": f"Wikipedia:{topic}",
                "domain": "Wikipedia",
                "language": language,
                "type": "encyclopedic"
            })
    except Exception as e:
        print(f"Error getting Wikipedia data: {e}")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    print("STARTING DATA COLLECTION FOR EquiNet")

    # Blog scraping
    for url in BLOG_URLS:
        scrape_blog(url)

    # Podcast transcription
    for i, url in enumerate(PODCAST_URLS):
        transcribe_podcast(url, i)

    # Wikipedia dumps
    for lang in LANGUAGES:
        get_wikipedia(lang)

    # Save dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Data collection complete. Dataset saved to {OUTPUT_FILE}")
