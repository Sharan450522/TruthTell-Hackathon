from typing import Dict, Any
from urllib.parse import urlparse
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def detect_platform(url: str) -> str:
    """Detect the platform from a given URL."""
    try:
        domain = urlparse(url).netloc.lower()

        platform_map = {
            'youtube.com': 'youtube',
            'youtu.be': 'youtube',
            'instagram.com': 'instagram',
            'facebook.com': 'facebook',
            'fb.com': 'facebook',
            'tiktok.com': 'tiktok',
            'twitter.com': 'twitter',
            'vimeo.com': 'vimeo'
        }

        for key, value in platform_map.items():
            if key in domain:
                return value

        return 'unknown'
    except Exception:
        raise ValueError('Invalid URL format')

def format_websocket_message(msg_type: str, data: Dict[str, Any]) -> str:
    """Format a message for WebSocket transmission."""
    return json.dumps({
        'type': msg_type,
        **data,
        'timestamp': datetime.now().isoformat()
    })

def safe_json_dumps(obj: Any) -> str:
    """Safely convert object to JSON string."""
    try:
        return json.dumps(obj)
    except Exception as e:
        logger.error(f'JSON serialization error: {str(e)}')
        return json.dumps({
            'error': 'Failed to serialize data',
            'timestamp': datetime.now().isoformat()
        })
