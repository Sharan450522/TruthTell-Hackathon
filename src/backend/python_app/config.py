import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
    ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
    GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
    PORT = int(os.getenv('PORT', 3000))

    NEWS_UPDATE_INTERVAL = 300  # 5 minutes
    CACHE_CLEANUP_INTERVAL = 3600  # 1 hour

    NEWS_SOURCES = {
        'RSS_FEEDS': [
            {
                'url': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms',
                'name': 'Times of India',
                'reliability': 0.8
            },
            {
                'url': 'https://www.thehindu.com/news/national/feeder/default.rss',
                'name': 'The Hindu',
                'reliability': 0.85
            }
        ],
        'GNEWS': {
            'endpoint': 'https://gnews.io/api/v4/top-headlines',
            'params': {
                'country': 'in',
                'lang': 'en',
                'max': 10,
                'token': GNEWS_API_KEY
            }
        }
    }

    CREDIBILITY_RULES = {
        'SUSPICIOUS_KEYWORDS': {
            'SENSATIONALISM': [
                {'word': 'shocking', 'weight': -8},
                {'word': 'unbelievable', 'weight': -7},
                {'word': 'sensational', 'weight': -6},
                {'word': 'breaking', 'weight': -5},
                {'word': 'exclusive', 'weight': -5}
            ],
            'CLICKBAIT': [
                {'word': "you won't believe", 'weight': -8},
                {'word': 'mind-blowing', 'weight': -7},
                {'word': 'viral', 'weight': -6},
                {'word': 'secret', 'weight': -5}
            ],
            'CONSPIRACY': [
                {'word': 'conspiracy', 'weight': -8},
                {'word': 'exposed', 'weight': -7},
                {'word': "they don't want you to know", 'weight': -8},
                {'word': 'hidden truth', 'weight': -7}
            ]
        },
        'CREDIBLE_INDICATORS': [
            {'phrase': 'according to research', 'weight': 5},
            {'phrase': 'studies show', 'weight': 4},
            {'phrase': 'experts say', 'weight': 3},
            {'phrase': 'official statement', 'weight': 5}
        ]
    }
