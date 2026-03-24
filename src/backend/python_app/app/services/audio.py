from typing import Optional, Dict, Any, Generator
import os
import tempfile
import wave
import audioop
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.temp_dir = tempfile.mkdtemp()
        self.active_processors: Dict[str, Any] = {}

    async def process_audio_chunk(self, chunk: bytes, session_id: str) -> Optional[bytes]:
        """Process an audio chunk and return normalized audio data."""
        try:
            # Check cache first
            cache_key = f"audio_chunk_{session_id}_{len(chunk)}"
            cached_result = self.cache_manager.get('audio', cache_key)
            if cached_result:
                return cached_result

            # Convert to mono if needed and normalize
            mono_data = await self._to_mono(chunk)
            normalized = await self._normalize_audio(mono_data)

            # Cache the result
            self.cache_manager.set('audio', cache_key, normalized, ttl=3600)
            return normalized

        except Exception as e:
            logger.error(f'Audio processing error: {str(e)}')
            return None

    async def _to_mono(self, audio_data: bytes) -> bytes:
        """Convert stereo audio to mono."""
        try:
            return audioop.tomono(audio_data, 2, 1, 1)
        except Exception:
            return audio_data

    async def _normalize_audio(self, audio_data: bytes) -> bytes:
        """Normalize audio volume."""
        try:
            return audioop.normalize(audio_data, 2, 65535)
        except Exception:
            return audio_data

    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        try:
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            logger.error(f'Cleanup error: {str(e)}')
