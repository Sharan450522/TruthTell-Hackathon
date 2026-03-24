from flask import Blueprint, request, jsonify
from flask_socketio import emit
from app.services.stream import StreamProcessor
from app.services.cache import CacheManager
from app.services.credibility import CredibilityAnalyzer
import logging

bp = Blueprint('stream', __name__)
logger = logging.getLogger(__name__)

@bp.route('/api/transcribe-recorded', methods=['POST'])
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
        analysis = CredibilityAnalyzer(CacheManager()).analyze(transcript.text, {
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

@bp.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    CacheManager().cleanup()
    return jsonify({'message': 'Cache cleared successfully'})

@bp.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'cache_stats': {
            'audio': len(CacheManager().caches.get('audio', {})),
            'news': len(CacheManager().caches.get('news', {})),
            'analysis': len(CacheManager().caches.get('analysis', {}))
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
            CredibilityAnalyzer(CacheManager())
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
