from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.serving import WSGIServer
import logging
import os

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

# Import routes
from app.routes import analysis, news, stream

# Register blueprints
app.register_blueprint(analysis.bp)
app.register_blueprint(news.bp)
app.register_blueprint(stream.bp)

# Initialize components
def initialize_app():
    from app.services.cache import CacheManager
    from app.services.credibility import CredibilityAnalyzer
    from app.services.audio import AudioProcessor
    from app.services.speech import SpeechProcessor
    from app.services.stream import StreamProcessor
    from app.services.news import NewsFetcher
    from app.utils.handlers import register_error_handlers

    # Initialize cache manager
    cache_manager = CacheManager()

    # Initialize credibility analyzer
    credibility_analyzer = CredibilityAnalyzer(cache_manager)

    # Initialize audio processor
    audio_processor = AudioProcessor(cache_manager)

    # Initialize speech processor
    speech_processor = SpeechProcessor(os.getenv('ASSEMBLYAI_API_KEY'), credibility_analyzer)

    # Initialize stream processor
    stream_processor = StreamProcessor(os.getenv('ASSEMBLYAI_API_KEY'), socketio, credibility_analyzer)

    # Initialize news fetcher
    news_fetcher = NewsFetcher(cache_manager, credibility_analyzer)

    # Register error handlers
    register_error_handlers(app)

    logger.info('Application components initialized successfully')

initialize_app()

if __name__ == '__main__':
    http_server = WSGIServer(('', int(os.getenv('PORT', 3000))), app)
    logger.info(f'Server running at http://localhost:{os.getenv("PORT", 3000)}')
    http_server.serve_forever()
