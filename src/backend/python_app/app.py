from flask import Flask, request, jsonify, Response, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import os
import threading
import time
import base64
import io
import subprocess
import shutil
import joblib
from datetime import datetime
import logging
import yt_dlp
import assemblyai as aai
import cv2
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from werkzeug.serving import WSGIServer
import signal
from typing import Optional, Dict, Any

# Import our modules
from config import Config
from app.services.cache import CacheManager
from app.services.credibility import CredibilityAnalyzer
from app.services.stream import StreamProcessor
from app.services.news import NewsFetcher
from app.utils.helpers import detect_platform
from video_processing import VideoProcessor

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components
cache_manager = CacheManager()
credibility_analyzer = CredibilityAnalyzer(cache_manager)
aai_client = aai.Client(Config.ASSEMBLYAI_API_KEY)
news_fetcher = NewsFetcher(cache_manager, credibility_analyzer)

# Active streams storage
active_streams: Dict[str, Any] = {}

# Optional local model artifacts for ensemble text analysis
MODEL_PATH = os.path.join("saved_models", "ensemble_model.pkl")
PIPELINE_PATH = os.path.join("saved_models", "pipeline.pkl")
ensemble_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
pipeline = joblib.load(PIPELINE_PATH) if os.path.exists(PIPELINE_PATH) else None

if ensemble_model is None or pipeline is None:
    logger.warning("saved_models artifacts missing; multimodal confidence falls back to credibility analyzer.")

video_processor = None


def analyze_text_with_ensemble(text: str) -> Dict[str, Any]:
    """Analyze text with saved ensemble model when available."""
    if not text or not text.strip():
        return {"prediction": "False", "confidence": 0.0}

    if ensemble_model is None or pipeline is None:
        analysis = credibility_analyzer.analyze(text, {"type": "fallback_credibility"})
        return {
            "prediction": "True" if analysis.get("is_credible") else "False",
            "confidence": float(analysis.get("confidence", 0.0)),
        }

    try:
        X = pipeline.transform([text])
        prediction = ensemble_model.predict(X)
        confidence = ensemble_model.predict_proba(X)[:, 1][0]
        return {
            "prediction": "True" if prediction[0] == 1 else "False",
            "confidence": float(confidence),
        }
    except Exception as e:
        logger.error(f"Ensemble analysis error: {e}")
        return {"prediction": "Error", "confidence": 0.0}


def scrape_article(url: str) -> Optional[str]:
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        article = soup.find("article") or soup.find("main") or soup.find("body")
        if article is None:
            return None
        paragraphs = [
            p.get_text(strip=True)
            for p in article.find_all("p")
            if len(p.get_text(strip=True)) > 30
        ]
        return " ".join(paragraphs)
    except Exception as e:
        logger.error(f"Error scraping article: {e}")
        return None


def ensure_video_processor() -> VideoProcessor:
    global video_processor
    if video_processor is None:
        video_processor = VideoProcessor(cache_manager, analyze_text_with_ensemble)
    return video_processor

def create_sse_response():
    """Create Server-Sent Events response"""
    return Response(
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text')
    language = data.get('language', 'en')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        analysis = credibility_analyzer.analyze(text, {
            'type': 'manual_input',
            'language': language
        })

        return jsonify({
            'text': text,
            'analysis': analysis,
            'success': True
        })
    except Exception as e:
        logger.error(f'Text analysis error: {str(e)}')
        return jsonify({
            'error': 'Failed to analyze text',
            'details': str(e)
        }), 500

@app.route('/api/transcribe-recorded', methods=['POST'])
def transcribe_recorded():
    data = request.get_json()
    video_url = data.get('video_url')
    language = data.get('language', 'en')
    ydl_process = None

    if not video_url:
        return jsonify({'error': 'Video URL is required'}), 400

    try:
        platform = detect_platform(video_url)
        logger.info(f'Starting transcription for {platform} video: {video_url}')

        # Download audio using yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'outtmpl': f'temp_{int(time.time())}.%(ext)s'
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            audio_file = ydl.prepare_filename(info).replace('.webm', '.mp3')

        # Upload to AssemblyAI
        with open(audio_file, 'rb') as f:
            upload_response = aai_client.upload(f)

        # Transcribe audio
        transcript = aai_client.transcribe(
            upload_response,
            config=aai.TranscriptionConfig(language_code=language)
        )

        # Analyze transcription
        analysis = credibility_analyzer.analyze(transcript.text, {
            'type': 'recorded_video',
            'platform': platform,
            'url': video_url
        })

        # Cleanup temporary file
        if os.path.exists(audio_file):
            os.remove(audio_file)

        return jsonify({
            'text': transcript.text,
            'platform': platform,
            'analysis': analysis,
            'success': True
        })

    except Exception as e:
        logger.error(f'Transcription Error: {str(e)}')
        return jsonify({
            'error': 'Failed to transcribe video',
            'details': str(e),
            'platform': detect_platform(video_url)
        }), 500


@app.route('/api/analyze-video', methods=['POST'])
def analyze_video_route():
    if "video" in request.files:
        video_file = request.files["video"]
        temp_path = os.path.join("temp", video_file.filename)
        os.makedirs("temp", exist_ok=True)
        video_file.save(temp_path)
        video_path = temp_path
    else:
        data = request.get_json() or {}
        video_path = data.get("file_path")
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "A valid file_path is required"}), 400

    try:
        result = ensure_video_processor().process_video(video_path)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/analyze-recorded-video', methods=['POST'])
def analyze_recorded_video_route():
    if "video" in request.files:
        video_file = request.files["video"]
        temp_path = os.path.join("temp", video_file.filename)
        os.makedirs("temp", exist_ok=True)
        video_file.save(temp_path)
        video_path = temp_path
    else:
        data = request.get_json() or {}
        video_path = data.get("file_path")

        if video_path and video_path.startswith("http"):
            temp_path = os.path.join("temp", "downloaded_video.mp4")
            os.makedirs("temp", exist_ok=True)
            try:
                cmd = [shutil.which("yt-dlp") or "yt-dlp", "-o", temp_path, "--recode-video", "mp4", video_path]
                subprocess.run(cmd, check=True)
                video_path = temp_path
            except Exception as e:
                return jsonify({"error": f"Video download failed: {e}"}), 400
        elif not video_path or not os.path.exists(video_path):
            return jsonify({"error": "A valid file_path is required"}), 400

    try:
        result = ensure_video_processor().process_recorded_video(video_path)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Recorded video analysis error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/analyze-article', methods=['POST'])
def analyze_article_route():
    data = request.get_json() or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "Article URL is required"}), 400

    article_text = scrape_article(url)
    if not article_text:
        return jsonify({"error": "Failed to extract article content"}), 500

    analysis = analyze_text_with_ensemble(article_text)
    return jsonify({"text": article_text, "analysis": analysis, "success": True})


@app.route('/api/analyze-media', methods=['POST'])
def analyze_media_route():
    if "media" not in request.files:
        return jsonify({"error": "Media file is required"}), 400

    media_file = request.files["media"]
    temp_path = os.path.join("temp", media_file.filename)
    os.makedirs("temp", exist_ok=True)
    media_file.save(temp_path)

    ext = os.path.splitext(media_file.filename)[1].lower()
    try:
        vp = ensure_video_processor()
        if ext in [".mp3", ".wav", ".m4a", ".aac"]:
            transcript = vp.transcribe_audio(temp_path)
            analysis = analyze_text_with_ensemble(transcript)
            result = {"transcript": transcript, "analysis": analysis, "media_type": "audio"}
        elif ext in [".mp4", ".mov", ".avi"]:
            result = vp.process_recorded_video(temp_path)
            result["media_type"] = "video"
        else:
            return jsonify({"error": "Unsupported file format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result), 200

@app.route('/api/news-stream')
def news_stream():
    def generate():
        while True:
            try:
                news = news_fetcher.fetch_trending_news()
                yield f'data: {json.dumps(news)}\n\n'
                time.sleep(Config.NEWS_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f'News stream error: {str(e)}')
                yield f'data: {json.dumps([])}\n\n'

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )

@app.route('/api/trending-news')
def trending_news():
    try:
        news = news_fetcher.fetch_trending_news()
        return jsonify(news)
    except Exception as e:
        logger.error(f'Error fetching trending news: {str(e)}')
        return jsonify({
            'error': 'Failed to fetch trending news',
            'details': str(e)
        }), 500

@app.route('/api/news-sources')
def news_sources():
    sources = [
        {'name': source['name'], 'url': source['url']}
        for source in Config.NEWS_SOURCES['RSS_FEEDS']
    ]
    sources.append({'name': 'GNews', 'url': 'https://gnews.io/'})
    return jsonify(sources)

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    cache_manager.cleanup()
    return jsonify({'message': 'Cache cleared successfully'})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'cache_stats': {
            'audio': len(cache_manager.caches.get('audio', {})),
            'news': len(cache_manager.caches.get('news', {})),
            'analysis': len(cache_manager.caches.get('analysis', {}))
        }
    })

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    logger.info('New WebSocket connection established')
    emit('status', {
        'type': 'status',
        'message': 'Connected to transcription service'
    })

@socketio.on('start_live')
def handle_start_live(data):
    try:
        url = data.get('url')
        if not url:
            raise ValueError('URL is required')

        platform = detect_platform(url)
        logger.info(f'Starting live stream processing for {platform}: {url}')

        # Kill existing stream if any
        stream_id = request.sid
        if stream_id in active_streams:
            active_streams[stream_id]['thread'].join(timeout=1)
            del active_streams[stream_id]

        # Initialize stream processor
        stream_processor = StreamProcessor(
            Config.ASSEMBLYAI_API_KEY,
            socketio,
            credibility_analyzer
        )

        # Start yt-dlp process
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }]
        }

        def process_stream():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        stream_thread = threading.Thread(target=process_stream)
        stream_thread.start()

        active_streams[stream_id] = {
            'thread': stream_thread,
            'processor': stream_processor,
            'platform': platform
        }

        emit('status', {
            'type': 'status',
            'message': f'Live stream processing started for {platform}',
            'platform': platform
        })

    except Exception as e:
        logger.error(f'WebSocket message error: {str(e)}')
        emit('error', {
            'type': 'error',
            'error': 'Failed to process message',
            'details': str(e)
        })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('WebSocket connection closed')
    stream_id = request.sid
    if stream_id in active_streams:
        active_streams[stream_id]['thread'].join(timeout=1)
        del active_streams[stream_id]

# Cleanup function
def cleanup():
    logger.info('Initiating cleanup...')
    for stream_id, stream_data in active_streams.items():
        stream_data['thread'].join(timeout=1)
    cache_manager.clear()

# Signal handlers
def signal_handler(signum, frame):
    cleanup()
    os._exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Periodic cache cleanup
def periodic_cleanup():
    while True:
        time.sleep(Config.CACHE_CLEANUP_INTERVAL)
        cache_manager.cleanup()

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    http_server = WSGIServer(('', Config.PORT), app)
    logger.info(f'Server running at http://localhost:{Config.PORT}')
    http_server.serve_forever()
