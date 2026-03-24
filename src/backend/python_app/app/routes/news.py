from flask import Blueprint, jsonify
from app.services.news import NewsFetcher
from app.services.cache import CacheManager
from app.services.credibility import CredibilityAnalyzer
import logging

bp = Blueprint('news', __name__)
logger = logging.getLogger(__name__)

@bp.route('/api/news-stream')
def news_stream():
    def generate():
        while True:
            try:
                news = NewsFetcher(CacheManager(), CredibilityAnalyzer(CacheManager())).fetch_trending_news()
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

@bp.route('/api/trending-news')
def trending_news():
    try:
        news = NewsFetcher(CacheManager(), CredibilityAnalyzer(CacheManager())).fetch_trending_news()
        return jsonify(news)
    except Exception as e:
        logger.error(f'Error fetching trending news: {str(e)}')
        return jsonify({
            'error': 'Failed to fetch trending news',
            'details': str(e)
        }), 500

@bp.route('/api/news-sources')
def news_sources():
    sources = [
        {'name': source['name'], 'url': source['url']}
        for source in Config.NEWS_SOURCES['RSS_FEEDS']
    ]
    sources.append({'name': 'GNews', 'url': 'https://gnews.io/'})
    return jsonify(sources)
