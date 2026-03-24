import asyncio
import aiohttp
from typing import List, Dict, Any
import feedparser
from datetime import datetime
import time
from string_similarity import stringsim
from config import Config

class NewsFetcher:
    def __init__(self, cache_manager, credibility_analyzer):
        self.cache_manager = cache_manager
        self.credibility_analyzer = credibility_analyzer

    async def fetch_gnews_articles(self) -> List[Dict[str, Any]]:
        try:
            if not Config.GNEWS_API_KEY:
                return []

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    Config.NEWS_SOURCES['GNEWS']['endpoint'],
                    params=Config.NEWS_SOURCES['GNEWS']['params']
                ) as response:
                    if response.ok:
                        data = await response.json()
                        return [{
                            'title': article['title'],
                            'description': article['description'],
                            'url': article['url'],
                            'source': article['source']['name'],
                            'published': article['publishedAt']
                        } for article in data.get('articles', [])]
            return []
        except Exception as e:
            logger.error(f'GNews API Error: {str(e)}')
            return []

    async def fetch_rss_feeds(self) -> List[Dict[str, Any]]:
        results = []
        for source in Config.NEWS_SOURCES['RSS_FEEDS']:
            try:
                feed = feedparser.parse(source['url'])
                items = [{
                    'title': entry.title,
                    'description': getattr(entry, 'description', '') or getattr(entry, 'summary', ''),
                    'url': entry.link,
                    'source': source['name'],
                    'reliability': source['reliability'],
                    'published': entry.published
                } for entry in feed.entries[:10]]
                results.extend(items)
            except Exception as e:
                logger.error(f'RSS feed error for {source["name"]}: {str(e)}')
        return results

    async def fetch_trending_news(self) -> List[Dict[str, Any]]:
        cached_news = self.cache_manager.get('news', 'trending')
        if cached_news:
            return cached_news

        try:
            # Fetch all news sources concurrently
            rss_results, gnews_results = await asyncio.gather(
                self.fetch_rss_feeds(),
                self.fetch_gnews_articles()
            )

            all_news = rss_results + gnews_results
            unique_news = []

            # De-duplicate and analyze news
            for current in all_news:
                is_duplicate = any(
                    stringsim.similarity(item['title'], current['title']) > 0.8
                    for item in unique_news
                )

                if not is_duplicate:
                    analysis = self.credibility_analyzer.analyze(
                        f"{current['title']} {current.get('description', '')}",
                        {
                            'source': current['source'],
                            'published': current['published']
                        }
                    )
                    unique_news.append({**current, 'analysis': analysis})

            # Sort by time and credibility
            sorted_news = sorted(
                unique_news,
                key=lambda x: (
                    -0.7 * x['analysis']['confidence']
                    - 0.3 * time.mktime(datetime.strptime(
                        x['published'],
                        '%a, %d %b %Y %H:%M:%S %z'
                    ).timetuple())
                )
            )

            result = sorted_news[:15]
            self.cache_manager.set('news', 'trending', result, Config.NEWS_UPDATE_INTERVAL)
            return result

        except Exception as e:
            logger.error(f'Error in fetch_trending_news: {str(e)}')
            return []
