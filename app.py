import io
import os
import re
import time
import json
import difflib
import requests
import feedparser
import logging
import subprocess
import threading
import base64
import cv2
import shutil
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import assemblyai as aai
from requests.exceptions import RequestException

# Import EnsembleModel if needed
from fake_news_predictor import FakeNewsPredictor
from feedback_store import PineconeTextStore, SQLiteFeedbackStore
from text_aggregator import aggregate_text, prepare_content_pieces

# Import VideoProcessor from video_processing.py (modified below)
from video_processing import VideoProcessor

# Load environment variables from .env file.
# Override is intentional so edits in .env replace stale shell/session values.
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask + Socket.IO initialization
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
socketio = SocketIO(app, cors_allowed_origins="*")

# Paths to saved models
MODEL_PATH = os.path.join('saved_models', 'ensemble_model.pkl')
PIPELINE_PATH = os.path.join('saved_models', 'pipeline.pkl')

MODEL_VERSION = os.getenv('MODEL_VERSION', 'ensemble_v1')
fake_news_predictor = FakeNewsPredictor(MODEL_PATH, PIPELINE_PATH, model_version=MODEL_VERSION)
feedback_store = SQLiteFeedbackStore(os.getenv('TRUTH_TELL_DB_PATH', os.path.join('data', 'truth_tell.db')))
pinecone_store = PineconeTextStore()

# AssemblyAI configuration
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
if not aai.settings.api_key:
    raise ValueError("ASSEMBLYAI_API_KEY is not set in environment variables")
# Instantiate transcriber using updated configuration
transcriber = aai.Transcriber(config=aai.TranscriptionConfig(language_code="en"))

# Paths to executables
YT_DLP_PATH = shutil.which("yt-dlp") or os.path.join(os.getcwd(), "yt-dlp.exe")
FFMPEG_PATH = shutil.which("ffmpeg") or os.path.join(os.getcwd(), "ffmpeg.exe")

# Verify executables exist
if not os.path.exists(YT_DLP_PATH):
    raise FileNotFoundError(f"yt-dlp not found at {YT_DLP_PATH}. Please install it or update the path.")
if not os.path.exists(FFMPEG_PATH):
    raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_PATH}. Please install it or update the path.")

# Simple Cache Manager for temporary in-memory storage
class CacheManager:
    def __init__(self):
        self.caches = {}
        self.initialize_caches()

    def initialize_caches(self):
        for cache_type in ['audio', 'news', 'analysis', 'liveStreams', 'video']:
            self.caches[cache_type] = {}

    def get(self, type, key):
        cache = self.caches.get(type, {})
        item = cache.get(key)
        if not item or (datetime.now() - item['timestamp']).total_seconds() > item['ttl']:
            return None
        return item['data']

    def set(self, type, key, data, ttl=300):
        if type not in self.caches:
            self.caches[type] = {}
        self.caches[type][key] = {
            'data': data,
            'timestamp': datetime.now(),
            'ttl': ttl
        }

    def delete(self, type, key):
        if type in self.caches and key in self.caches[type]:
            del self.caches[type][key]

    def clear(self):
        self.caches = {}
        self.initialize_caches()

cache_manager = CacheManager()

def classify_text_only(text, source_type=None):
    return fake_news_predictor.predict(text, source_type=source_type)


def needs_evidence_verification(analysis, source_type):
    return (
        source_type == 'live_stream'
        or analysis.get('label') == 'uncertain'
        or float(analysis.get('confidence', 0.0)) < 0.55
    )


def analyze_content(pieces, source_type, url=None, file_name=None):
    prepared_pieces = prepare_content_pieces(pieces)
    merged_text = aggregate_text(prepared_pieces, source_type=source_type)

    try:
        source_id = feedback_store.create_source(source_type, url=url, file_name=file_name)
        feedback_store.save_extracted_content(source_id, prepared_pieces)
        analysis = fake_news_predictor.predict(merged_text, source_type=source_type)
        analysis['needs_verification'] = needs_evidence_verification(analysis, source_type)
        prediction_id = feedback_store.save_prediction(source_id, analysis, merged_text)
        vector_result = pinecone_store.upsert_content(
            source_id,
            source_type,
            prepared_pieces,
            prediction_id=prediction_id,
        )
        logger.info(
            "Stored analysis: source_id=%s prediction_id=%s source_type=%s label=%s confidence=%.4f vector_storage=%s",
            source_id,
            prediction_id,
            source_type,
            analysis.get('label'),
            float(analysis.get('confidence', 0.0)),
            vector_result,
        )
        analysis.update({
            'source_id': source_id,
            'prediction_id': prediction_id,
            'vector_storage': vector_result,
        })
    except Exception as e:
        logger.error(f'Error storing analysis result: {e}')
        analysis.update({
            'source_id': None,
            'prediction_id': None,
            'vector_storage': {'enabled': False, 'upserted': 0, 'error': str(e)},
        })

    return analysis, merged_text


def analyze_text(text, source_type='text', url=None, file_name=None):
    pieces = [{'type': 'text', 'text': text}]
    analysis, _ = analyze_content(pieces, source_type, url=url, file_name=file_name)
    return analysis


def analyze_text_without_storage(text, source_type='text'):
    analysis = fake_news_predictor.predict(text, source_type=source_type)
    analysis['needs_verification'] = needs_evidence_verification(analysis, source_type)
    analysis.update({
        'source_id': None,
        'prediction_id': None,
        'vector_storage': {'enabled': False, 'upserted': 0, 'skipped': True},
    })
    return analysis

# Instantiate VideoProcessor with a classifier-only function; routes persist the final
# aggregated transcript/OCR result to avoid duplicate DB rows.
video_processor = VideoProcessor(cache_manager, classify_text_only)

# Helper function: scrape article content
def scrape_article(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'header', 'footer', '.ads', '#comments']):
            element.decompose()
        article = soup.find('article') or soup.find('main') or soup.find('body')
        paragraphs = [p.get_text(strip=True) for p in article.find_all('p') if len(p.get_text(strip=True)) > 30]
        return ' '.join(paragraphs)
    except Exception as e:
        logger.error(f'Error scraping article: {e}')
        return None

# News sources configuration (RSS and GNews)
NEWS_SOURCES = {
    'RSS_FEEDS': [
        {'url': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms', 'name': 'Times of India', 'reliability': 0.8},
        {'url': 'https://www.thehindu.com/news/national/feeder/default.rss', 'name': 'The Hindu', 'reliability': 0.85}
    ],
    'GNEWS': {
        'endpoint': 'https://gnews.io/api/v4/top-headlines',
        'params': {
            'country': 'in',
            'lang': 'en',
            'max': 10,
            'token': os.getenv('GNEWS_API_KEY')
        }
    }
}

def fetch_gnews_articles():
    if os.getenv('ENABLE_GNEWS', '').strip().lower() not in {'1', 'true', 'yes'}:
        logger.info('GNews fetch skipped. Set ENABLE_GNEWS=true to enable it.')
        return []
    api_key = os.getenv('GNEWS_API_KEY')
    if not api_key:
        logger.warning('GNEWS_API_KEY is not set')
        return []
    try:
        response = requests.get(NEWS_SOURCES['GNEWS']['endpoint'], params=NEWS_SOURCES['GNEWS']['params'], timeout=10)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        return articles
    except RequestException as e:
        logger.error(f'GNews API error: {e}')
        return []

def fetch_trending_news():
    cached_news = cache_manager.get('news', 'trending')
    if cached_news:
        return cached_news
    try:
        rss_results = []
        for source in NEWS_SOURCES['RSS_FEEDS']:
            feed = feedparser.parse(source['url'])
            for item in feed.entries[:10]:
                rss_results.append({
                    'title': item.title,
                    'description': item.get('description', item.get('summary', '')),
                    'url': item.link,
                    'source': source['name'],
                    'reliability': source['reliability'],
                    'published': item.get('published', '')
                })
        gnews_results = fetch_gnews_articles()
        all_news = rss_results + gnews_results
        unique_news = []
        for current in all_news:
            is_duplicate = any(difflib.SequenceMatcher(None, item['title'], current['title']).ratio() > 0.8 for item in unique_news)
            if not is_duplicate:
                text = f"{current['title']} {current.get('description', '')}"
                analysis = analyze_text_without_storage(text, source_type='trending_news')
                unique_news.append({**current, 'analysis': analysis})
        result = unique_news[:15]
        cache_manager.set('news', 'trending', result)
        return result
    except Exception as e:
        logger.error(f'Error fetching trending news: {e}')
        return []

# ---------------------------------------------------------------------------
# Helper: Upload audio buffer to AssemblyAI
# ---------------------------------------------------------------------------
def upload_audio_buffer(audio_buffer):
    upload_endpoint = "https://api.assemblyai.com/v2/upload"
    headers = {"authorization": aai.settings.api_key}
    audio_buffer.seek(0)
    response = requests.post(upload_endpoint, headers=headers, data=audio_buffer)
    response.raise_for_status()
    upload_url = response.json()['upload_url']
    return upload_url

# ---------------------------------------------------------------------------
# Recorded Transcription Endpoint (for audio-only transcription)
# ---------------------------------------------------------------------------
@app.route('/api/transcribe-recorded', methods=['POST'])
def transcribe_recorded_route():
    data = request.get_json()
    video_url = data.get('video_url')
    if not video_url:
        return jsonify({"error": "Video URL is required"}), 400

    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to extract audio from {video_url} (Attempt {attempt + 1}/{max_retries})")
            process = subprocess.Popen(
                [YT_DLP_PATH, '-x', '--audio-format', 'mp3', '--output', '-', '--no-playlist', video_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            audio_data, error = process.communicate(timeout=120)
            if process.returncode != 0:
                error_msg = error.decode()
                logger.error(f'yt-dlp failed: {error_msg}')
                raise Exception(f'yt-dlp failed: {error_msg}')
            logger.info(f"Audio extracted: {len(audio_data)} bytes")

            audio_buffer = io.BytesIO(audio_data)
            cache_key = f"recorded_{int(time.time() * 1000)}"
            cache_manager.set('audio', cache_key, audio_buffer, ttl=60)

            cached_audio = cache_manager.get('audio', cache_key)
            if not cached_audio:
                raise Exception("Failed to retrieve audio from cache")
            logger.info("Uploading audio to AssemblyAI using in-memory buffer")
            upload_url = upload_audio_buffer(cached_audio)
            logger.info("Transcribing audio")
            transcript = transcriber.transcribe(upload_url)
            if transcript.error:
                logger.error(f"Transcription error: {transcript.error}")
                raise Exception(f"Transcription error: {transcript.error}")
            if not transcript.text:
                raise Exception("Transcription returned empty text")
            logger.info("Transcription successful")

            cache_manager.delete('audio', cache_key)

            analysis = analyze_text(
                transcript.text,
                source_type='youtube_recorded_video',
                url=video_url,
            )
            return jsonify({'text': transcript.text, 'analysis': analysis, 'success': True})

        except subprocess.TimeoutExpired:
            process.terminate()
            logger.error(f"Transcription timed out on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return jsonify({"error": "Transcription timed out after retries"}), 500

        except Exception as e:
            logger.error(f'Transcription error on attempt {attempt + 1}/{max_retries}: {e}')
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

# ---------------------------------------------------------------------------
# Live Stream Processing
# ---------------------------------------------------------------------------
def emit_video_fragment(sid, frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        raise ValueError("Failed to encode frame.")
    fragment_base64 = base64.b64encode(buffer).decode('utf-8')
    socketio.emit('video_fragment', {
        'fragment': fragment_base64,
        'timestamp': datetime.now().isoformat(),
        'label': 'Live Frame'
    }, room=sid)

def build_ytdlp_command(format_selector, output_path, video_url, extra_args=None):
    cmd = [
        YT_DLP_PATH,
        '--no-playlist',
        '--retries', '10',
        '--fragment-retries', '10',
        '--concurrent-fragments', '1',
        '--hls-use-mpegts',
        '--no-part',
        '--force-overwrites',
        '-f', format_selector,
    ]
    js_runtime = os.getenv('YT_DLP_JS_RUNTIME', '').strip()
    if js_runtime:
        cmd.extend(['--js-runtimes', js_runtime])

    cookies_file = os.getenv('YT_DLP_COOKIES_FILE', '').strip()
    if cookies_file:
        cmd.extend(['--cookies', cookies_file])

    cookies_from_browser = os.getenv('YT_DLP_COOKIES_FROM_BROWSER', '').strip()
    use_browser_cookies_for_live = os.getenv('YT_DLP_USE_BROWSER_COOKIES_FOR_LIVE', '').strip().lower() in {
        '1', 'true', 'yes'
    }
    if cookies_from_browser and use_browser_cookies_for_live:
        cmd.extend(['--cookies-from-browser', cookies_from_browser])

    extractor_args = os.getenv('YT_DLP_EXTRACTOR_ARGS', '').strip()
    if extractor_args:
        cmd.extend(['--extractor-args', extractor_args])

    if extra_args:
        cmd.extend(extra_args)

    cmd.extend(['-o', output_path, video_url])
    return cmd


def capture_live_audio_segment(video_url, temp_audio, segment_duration):
    audio_format = os.getenv('YT_DLP_LIVE_AUDIO_FORMAT', 'ba/bestaudio/best')
    captured_error_logs = []
    min_valid_size = 10_000

    cmd_variants = [
        build_ytdlp_command(audio_format, temp_audio, video_url),
        build_ytdlp_command('bestaudio/best', temp_audio, video_url, ['--extractor-args', 'youtube:player_client=tv']),
    ]

    for idx, cmd in enumerate(cmd_variants, start=1):
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        timed_out = False
        try:
            _, error = process.communicate(timeout=segment_duration + 20)
        except subprocess.TimeoutExpired:
            process.terminate()
            _, error = process.communicate(timeout=15)
            timed_out = True
            logger.info(f"Stopped yt-dlp audio capture after live window (variant {idx})")

        error_msg = error.decode(errors='ignore')
        file_exists = os.path.exists(temp_audio)
        file_size = os.path.getsize(temp_audio) if file_exists else 0
        valid_audio = file_exists and file_size >= min_valid_size

        if valid_audio and (process.returncode == 0 or timed_out):
            with open(temp_audio, "rb") as f:
                return f.read()

        if "Could not copy Chrome cookie database" in error_msg:
            logger.warning("Skipping cookies-from-browser due to locked Chrome cookie DB.")
        else:
            logger.warning(f"yt-dlp live audio capture attempt {idx} failed")

        captured_error_logs.append(error_msg[:1200])

    ffmpeg_error = capture_live_audio_with_stream_url(video_url, temp_audio, segment_duration)
    if ffmpeg_error is None and os.path.exists(temp_audio) and os.path.getsize(temp_audio) >= min_valid_size:
        with open(temp_audio, "rb") as f:
            return f.read()
    if ffmpeg_error:
        captured_error_logs.append(ffmpeg_error[:1200])

    raise Exception(
        "Live audio capture failed after fallbacks. "
        + " | ".join([msg for msg in captured_error_logs if msg])
    )


def capture_live_audio_with_stream_url(video_url, temp_audio, segment_duration):
    """
    Resolve a direct audio URL with yt-dlp, then let ffmpeg capture a short
    audio-only segment for transcription.
    """
    stream_format = os.getenv('YT_DLP_LIVE_AUDIO_FORMAT', 'ba/bestaudio/best')
    url_cmd = [
        YT_DLP_PATH,
        '--no-playlist',
        '--no-warnings',
        '-f', stream_format,
        '--get-url',
        video_url,
    ]
    js_runtime = os.getenv('YT_DLP_JS_RUNTIME', '').strip()
    if js_runtime:
        url_cmd[1:1] = ['--js-runtimes', js_runtime]

    try:
        url_process = subprocess.run(
            url_cmd,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if url_process.returncode != 0:
            return url_process.stderr

        urls = [line.strip() for line in url_process.stdout.splitlines() if line.strip()]
        if not urls:
            return "yt-dlp did not return a playable audio stream URL"

        ffmpeg_cmd = [
            FFMPEG_PATH,
            '-y',
            '-loglevel', 'error',
            '-t', str(segment_duration),
            '-i', urls[-1],
            '-vn',
            '-acodec', 'libmp3lame',
            '-ar', '16000',
            '-ac', '1',
            temp_audio,
        ]

        ffmpeg_process = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=segment_duration + 30,
            check=False,
        )
        if ffmpeg_process.returncode != 0:
            return ffmpeg_process.stderr
        logger.info("Captured live audio with ffmpeg stream URL fallback")
        return None
    except Exception as e:
        return str(e)


def emit_live_video_preview(video_url, temp_fragment, sid, segment_duration):
    try:
        video_format = os.getenv('YT_DLP_LIVE_VIDEO_FORMAT', 'bv*[height<=480]/best[height<=480]/best')
        cmd = build_ytdlp_command(video_format, temp_fragment, video_url)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            _, _ = process.communicate(timeout=min(segment_duration, 12))
        except subprocess.TimeoutExpired:
            process.terminate()
            _, _ = process.communicate(timeout=10)

        if not os.path.exists(temp_fragment) or os.path.getsize(temp_fragment) < 10_000:
            return

        cap = cv2.VideoCapture(temp_fragment)
        ret, frame = cap.read()
        cap.release()
        if ret:
            emit_video_fragment(sid, frame)
    except Exception as e:
        logger.info(f"Live video preview skipped: {e}")
    finally:
        if os.path.exists(temp_fragment):
            os.remove(temp_fragment)


def recorded_video_download_commands(video_url, temp_path):
    recorded_format = os.getenv('YT_DLP_RECORDED_FORMAT', 'bv*+ba/best')
    base_cmd = [
        YT_DLP_PATH,
        '--no-playlist',
        '--force-overwrites',
        '--retries', '10',
        '--fragment-retries', '10',
        '-f', recorded_format,
        '--merge-output-format', 'mp4',
        '--recode-video', 'mp4',
    ]

    js_runtime = os.getenv('YT_DLP_JS_RUNTIME', '').strip()
    if js_runtime:
        base_cmd.extend(['--js-runtimes', js_runtime])

    commands = []
    cookies_file = os.getenv('YT_DLP_COOKIES_FILE', '').strip()
    if cookies_file:
        commands.append(base_cmd + ['--cookies', cookies_file, '-o', temp_path, video_url])

    commands.append(base_cmd + ['-o', temp_path, video_url])

    cookies_from_browser = os.getenv('YT_DLP_COOKIES_FROM_BROWSER', '').strip()
    recorded_cookie_setting = os.getenv('YT_DLP_USE_BROWSER_COOKIES_FOR_RECORDED', '').strip().lower()
    use_browser_cookies = recorded_cookie_setting in {'1', 'true', 'yes'} or (
        bool(cookies_from_browser) and recorded_cookie_setting not in {'0', 'false', 'no'}
    )
    if cookies_from_browser and use_browser_cookies:
        commands.append(base_cmd + ['--cookies-from-browser', cookies_from_browser, '-o', temp_path, video_url])

    return commands


def ytdlp_config_status():
    cookies_file = os.getenv('YT_DLP_COOKIES_FILE', '').strip()
    cookies_file_exists = bool(cookies_file and os.path.exists(cookies_file))
    cookies_file_size = os.path.getsize(cookies_file) if cookies_file_exists else 0
    return {
        "js_runtime": os.getenv('YT_DLP_JS_RUNTIME', '').strip(),
        "recorded_format": os.getenv('YT_DLP_RECORDED_FORMAT', '').strip(),
        "cookies_file_configured": bool(cookies_file),
        "cookies_file_exists": cookies_file_exists,
        "cookies_file_size": cookies_file_size,
        "cookies_from_browser_configured": bool(os.getenv('YT_DLP_COOKIES_FROM_BROWSER', '').strip()),
        "use_browser_cookies_for_recorded": os.getenv('YT_DLP_USE_BROWSER_COOKIES_FOR_RECORDED', '').strip(),
        "use_browser_cookies_for_live": os.getenv('YT_DLP_USE_BROWSER_COOKIES_FOR_LIVE', '').strip(),
    }


def download_recorded_video(video_url, temp_path):
    captured_errors = []
    logger.info("yt-dlp recorded config: %s", ytdlp_config_status())
    for index, cmd in enumerate(recorded_video_download_commands(video_url, temp_path), start=1):
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.info(f"Downloading recorded video from URL using yt-dlp variant {index}: {video_url}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and os.path.exists(temp_path):
            return temp_path
        captured_errors.append((result.stderr or result.stdout or '')[-1600:])

    raise RuntimeError("Video download failed after fallbacks. " + " | ".join(captured_errors))


def summarize_download_error(error_text):
    lowered = str(error_text).lower()
    if "sign in to confirm" in lowered or "not a bot" in lowered:
        return (
            "YouTube requires authenticated cookies for this URL. Export cookies to cookies.txt "
            "or enable browser cookies for recorded downloads."
        )
    if "no supported javascript runtime" in lowered:
        return "yt-dlp needs a JavaScript runtime. Install Deno and set YT_DLP_JS_RUNTIME=deno."
    if "could not copy chrome cookie database" in lowered:
        return "Chrome cookie database is locked. Close Chrome or use an exported cookies.txt file."
    if "requested format is not available" in lowered:
        return "Requested YouTube format is unavailable. Try YT_DLP_RECORDED_FORMAT=best."
    return "Recorded video download failed. Check yt-dlp cookies/runtime/format settings."

def start_live_stream(sid, video_url, stop_event):
    max_retries = 3
    segment_duration = 30  # seconds
    # Initialize variables for cumulative confidence analysis
    prev_conf = None
    alpha = 0.5  # Smoothing factor (adjust as needed)

    while not stop_event.is_set():
        for attempt in range(max_retries):
            try:
                # Download a short live segment using yt-dlp directly.
                # This avoids handing ffmpeg an expiring manifest URL that can trigger 403 errors.
                os.makedirs('temp', exist_ok=True)
                temp_fragment = os.path.join('temp', f"live_segment_{sid}.mp4")
                temp_audio = os.path.join('temp', f"live_audio_{sid}.mp3")

                video_data = capture_live_audio_segment(video_url, temp_audio, segment_duration)
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)

                # Frame extraction is helpful for the UI, but transcription should
                # not fail just because YouTube exposes no usable video-only format.
                emit_live_video_preview(video_url, temp_fragment, sid, segment_duration)
                
                # Process the audio segment
                audio_buffer = io.BytesIO(video_data)
                cache_key = f"live_{int(time.time() * 1000)}"
                cache_manager.set('audio', cache_key, audio_buffer, ttl=60)
                cached_audio = cache_manager.get('audio', cache_key)
                if not cached_audio:
                    raise Exception("Failed to retrieve live audio from cache")
                logger.info("Uploading live audio to AssemblyAI using in-memory buffer")
                upload_url = upload_audio_buffer(cached_audio)
                transcript = transcriber.transcribe(upload_url)
                if transcript.error:
                    logger.error(f"Transcription error: {transcript.error}")
                    raise Exception(f"Transcription error: {transcript.error}")
                if not transcript.text:
                    logger.warning("Transcription returned empty text")
                    continue

                cache_manager.delete('audio', cache_key)
                # Analyze the current segment's transcript
                analysis = analyze_text(
                    transcript.text,
                    source_type='live_stream',
                    url=video_url,
                )
                curr_conf = analysis.get('confidence', 0.0)

                # Update cumulative confidence using exponential smoothing
                if prev_conf is None:
                    new_conf = curr_conf
                else:
                    new_conf = alpha * curr_conf + (1 - alpha) * prev_conf
                prev_conf = new_conf

                # Emit transcription along with the cumulative confidence analysis
                socketio.emit('transcription', {
                    'text': transcript.text,
                    'analysis': {
                        **analysis,
                        'cumulative_confidence': new_conf
                    },
                    'timestamp': datetime.now().isoformat(),
                    'type': 'Segment'
                }, room=sid)
                break  # Exit the retry loop on success

            except Exception as e:
                logger.error(f"Error in live stream processing (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    socketio.emit('error', {'error': str(e)}, room=sid)
        time.sleep(segment_duration)

# ---------------------------------------------------------------------------
# Endpoint: Video Analysis (standard)
# ---------------------------------------------------------------------------
@app.route('/api/analyze-video', methods=['POST'])
def analyze_video_route():
    if 'video' in request.files:
        video_file = request.files['video']
        temp_path = os.path.join('temp', video_file.filename)
        os.makedirs('temp', exist_ok=True)
        video_file.save(temp_path)
        video_path = temp_path
        file_name = video_file.filename
        logger.info(f"Received video file via form-data, saved to {temp_path}")
    else:
        data = request.get_json()
        video_path = data.get('file_path')
        file_name = os.path.basename(video_path) if video_path else None
        logger.info(f"Received video file path: {video_path}")
        if not video_path or not os.path.exists(video_path):
            logger.error("A valid file_path is required.")
            return jsonify({"error": "A valid file_path is required"}), 400
    try:
        logger.info("Calling video_processor.process_video...")
        result = video_processor.process_video(video_path)
        pieces = [
            {'type': 'transcript', 'text': result.get('transcript', '')},
            {'type': 'frame_ocr', 'text': result.get('ocr_text', '')},
        ]
        analysis, classification_text = analyze_content(
            pieces,
            source_type='video_file',
            file_name=file_name,
        )
        result['analysis'] = analysis
        result['classification_text'] = classification_text
        logger.info("Video analysis complete. Analysis summary:")
        for key, value in result.items():
            logger.info(f"{key}: {value}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Temporary file {temp_path} removed.")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------------
# Endpoint: Recorded Video Analysis (full pipeline)
# ---------------------------------------------------------------------------
@app.route('/api/analyze-recorded-video', methods=['POST'])
def analyze_recorded_video_route():
    source_url = None
    file_name = None
    if 'video' in request.files:
        video_file = request.files['video']
        temp_path = os.path.join('temp', video_file.filename)
        os.makedirs('temp', exist_ok=True)
        video_file.save(temp_path)
        video_path = temp_path
        file_name = video_file.filename
        logger.info(f"Received recorded video via form-data, saved to {temp_path}")
    else:
        data = request.get_json()
        video_path = data.get('file_path')
        file_name = os.path.basename(video_path) if video_path else None
        logger.info(f"Received recorded video file path: {video_path}")
        if video_path and video_path.startswith("http"):
            source_url = video_path
            temp_path = os.path.join('temp', "downloaded_video.mp4")
            file_name = "downloaded_video.mp4"
            os.makedirs('temp', exist_ok=True)
            try:
                video_path = download_recorded_video(video_path, temp_path)
            except Exception as e:
                logger.error(f"Video download failed: {e}")
                return jsonify({
                    "error": summarize_download_error(str(e)),
                    "details": str(e)[-4000:],
                    "success": False,
                }), 400

        elif not video_path or not os.path.exists(video_path):
            logger.error("A valid file_path is required for recorded video analysis.")
            return jsonify({"error": "A valid file_path is required"}), 400
    try:
        logger.info("Calling video_processor.process_recorded_video...")
        result = video_processor.process_recorded_video(video_path)
        pieces = [
            {'type': 'transcript', 'text': result.get('transcript', '')},
            {'type': 'frame_ocr', 'text': result.get('video_summary', {}).get('ocr_text', '')},
        ]
        analysis, classification_text = analyze_content(
            pieces,
            source_type='recorded_video',
            url=source_url,
            file_name=file_name,
        )
        result['video_summary']['analysis'] = analysis
        result['video_summary']['classification_text'] = classification_text
        result['analysis'] = analysis
        logger.info("Recorded video analysis complete. Analysis summary:")
        for key, value in result['video_summary'].items():
            logger.info(f"{key}: {value}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Temporary file {temp_path} removed.")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Recorded video analysis error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/test-recorded-download', methods=['POST'])
def test_recorded_download_route():
    data = request.get_json() or {}
    video_url = data.get('url') or data.get('file_path')
    if not video_url:
        return jsonify({"error": "url is required"}), 400
    if not video_url.startswith("http"):
        return jsonify({"error": "url must be an http(s) video URL"}), 400

    os.makedirs('temp', exist_ok=True)
    temp_path = os.path.join('temp', "download_test_video.mp4")
    try:
        download_recorded_video(video_url, temp_path)
        file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
        return jsonify({
            "success": True,
            "file_size": file_size,
            "message": "Recorded video download succeeded.",
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": summarize_download_error(str(e)),
            "details": str(e)[-4000:],
        }), 400
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---------------------------------------------------------------------------
# Other Endpoints and Socket.IO Events
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze-article', methods=['POST'])
def analyze_article_route():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "Article URL is required"}), 400
    article_text = scrape_article(url)
    if not article_text:
        return jsonify({"error": "Failed to extract article content"}), 500
    analysis, classification_text = analyze_content(
        [{'type': 'article_text', 'text': article_text}],
        source_type='article',
        url=url,
    )
    return jsonify({
        'text': article_text,
        'classification_text': classification_text,
        'analysis': analysis,
        'success': True,
    })



@app.route('/api/analyze-media', methods=['POST'])
def analyze_media_route():
    if 'media' not in request.files:
        return jsonify({"error": "Media file is required"}), 400

    media_file = request.files['media']
    temp_path = os.path.join('temp', media_file.filename)
    os.makedirs('temp', exist_ok=True)
    media_file.save(temp_path)

    ext = os.path.splitext(media_file.filename)[1].lower()
    try:
        if ext in ['.mp3', '.wav', '.m4a', '.aac']:
            # Process as an audio file
            transcript = video_processor.transcribe_audio(temp_path)
            analysis = analyze_text(
                transcript,
                source_type='audio_file',
                file_name=media_file.filename,
            )
            result = {
                'transcript': transcript,
                'analysis': analysis,
                'media_type': 'audio'
            }
        elif ext in ['.mp4', '.mov', '.avi']:
            # Process as a video file using the recorded video pipeline
            result = video_processor.process_recorded_video(temp_path)
            pieces = [
                {'type': 'transcript', 'text': result.get('transcript', '')},
                {'type': 'frame_ocr', 'text': result.get('video_summary', {}).get('ocr_text', '')},
            ]
            analysis, classification_text = analyze_content(
                pieces,
                source_type='video_file',
                file_name=media_file.filename,
            )
            result['video_summary']['analysis'] = analysis
            result['video_summary']['classification_text'] = classification_text
            result['analysis'] = analysis
            result['media_type'] = 'video'
        else:
            return jsonify({"error": "Unsupported file format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return jsonify(result), 200


@app.route('/api/analyze-text', methods=['POST'])
def analyze_text_route():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    analysis = analyze_text(text, source_type='text')
    return jsonify({'text': text, 'analysis': analysis, 'success': True})


@app.route('/api/review-label', methods=['POST'])
def review_label_route():
    data = request.get_json() or {}
    source_id = data.get('source_id')
    prediction_id = data.get('prediction_id')
    human_label = data.get('human_label')
    notes = data.get('notes')

    if not source_id or not prediction_id or not human_label:
        return jsonify({
            "error": "source_id, prediction_id, and human_label are required"
        }), 400

    try:
        review = feedback_store.save_review_label(source_id, prediction_id, human_label, notes=notes)
        return jsonify({'success': True, 'review': review}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error saving review label: {e}")
        return jsonify({"error": "Failed to save review label"}), 500


@app.route('/api/vector-status', methods=['GET'])
def vector_status_route():
    return jsonify(pinecone_store.status())


@app.route('/api/ytdlp-status', methods=['GET'])
def ytdlp_status_route():
    return jsonify(ytdlp_config_status())

@app.route('/api/trending-news', methods=['GET'])
def trending_news_route():
    return jsonify(fetch_trending_news())

@app.route('/api/news-stream')
def news_stream_route():
    def generate():
        while True:
            news = fetch_trending_news()
            yield f"data: {json.dumps(news)}\n\n"
            time.sleep(10)
    return app.response_class(generate(), mimetype='text/event-stream')

active_streams = {}

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in active_streams:
        active_streams[sid]['stop_event'].set()
        active_streams[sid]['thread'].join()
        del active_streams[sid]
    logger.info('Client disconnected')

@socketio.on('start_live')
def handle_start_live(data):
    video_url = data.get('url')
    sid = request.sid
    if not video_url:
        socketio.emit('error', {'error': 'Video URL is required'}, room=sid)
        return

    if sid in active_streams:
        active_streams[sid]['stop_event'].set()
        active_streams[sid]['thread'].join()
        del active_streams[sid]

    stop_event = threading.Event()
    thread = threading.Thread(target=start_live_stream, args=(sid, video_url, stop_event))
    thread.start()
    active_streams[sid] = {'thread': thread, 'stop_event': stop_event}
    socketio.emit('status', {'message': f"Started live stream for {video_url}"}, room=sid)

@socketio.on('stop_live')
def handle_stop_live():
    sid = request.sid
    if sid in active_streams:
        active_streams[sid]['stop_event'].set()
        active_streams[sid]['thread'].join()
        del active_streams[sid]
        socketio.emit('status', {'message': 'Live stream stopped'}, room=sid)
    else:
        socketio.emit('error', {'error': 'No active stream to stop'}, room=sid)

if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 3000))
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False, use_reloader=False)
