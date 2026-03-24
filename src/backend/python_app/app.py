from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import os
import threading
import time
from datetime import datetime
import logging
import yt_dlp
import assemblyai as aai
from werkzeug.serving import WSGIServer
import signal
from typing import Optional, Dict, Any

# Import our modules
from config import Config
from app.services.cache import CacheManager
from app.services.credibility import CredibilityAnalyzer
from app.services.stream import StreamProcessor

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

# Active streams storage
active_streams: Dict[str, Any] = {}

def create_sse_response():
    """Create Server-Sent Events response"""
    return Response(
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )

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

@app.route('/api/news-stream')
def news_stream():
    def generate():
        while True:
            try:
                news = fetch_trending_news()
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
        news = fetch_trending_news()
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
        success = live_stream_handler.start_stream(stream_id, data.get('url'))
        if success:
            emit('status', {'message': 'Stream started successfully'})
        else:
            emit('error', {'message': 'Failed to start stream'})
        if stream_id in active_streams:
            active_streams[stream_id]['process'].terminate()

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
