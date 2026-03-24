from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import threading

class CacheManager:
    def __init__(self):
        self.caches: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.initialize_caches()

    def initialize_caches(self):
        cache_types = ['audio', 'news', 'analysis', 'liveStreams']
        for cache_type in cache_types:
            self.caches[cache_type] = {}

    def get(self, cache_type: str, key: str) -> Optional[Any]:
        with self._lock:
            cache = self.caches.get(cache_type, {})
            item = cache.get(key)
            if item and not self._is_expired(item['timestamp'], item['ttl']):
                return item['data']
            return None

    def set(self, cache_type: str, key: str, data: Any, ttl: int = 3600) -> None:
        with self._lock:
            if cache_type not in self.caches:
                self.caches[cache_type] = {}
            self.caches[cache_type][key] = {
                'data': data,
                'timestamp': datetime.now(),
                'ttl': ttl
            }

    def delete(self, cache_type: str, key: str) -> None:
        with self._lock:
            if cache_type in self.caches:
                self.caches[cache_type].pop(key, None)

    def clear(self, cache_type: Optional[str] = None) -> None:
        with self._lock:
            if cache_type:
                self.caches[cache_type] = {}
            else:
                for cache in self.caches.values():
                    cache.clear()

    def _is_expired(self, timestamp: datetime, ttl: int) -> bool:
        return (datetime.now() - timestamp) > timedelta(seconds=ttl)

    def cleanup(self) -> None:
        with self._lock:
            for cache_type, cache in self.caches.items():
                expired_keys = [
                    key for key, value in cache.items()
                    if self._is_expired(value['timestamp'], value['ttl'])
                ]
                for key in expired_keys:
                    self.delete(cache_type, key)
