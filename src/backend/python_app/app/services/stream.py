import threading
from queue import Queue
import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class StreamProcessor:
    def __init__(self, api_key: str, websocket: Any, analyzer: CredibilityAnalyzer):
        self.client = aai.Client(api_key)
        self.websocket = websocket
        self.analyzer = analyzer
        self.buffer = io.BytesIO()
        self.is_processing = False
        self.lock = threading.Lock()

    def process_chunk(self, chunk: bytes, duration: float) -> None:
        with self.lock:
            self.buffer.write(chunk)

            if self.buffer.tell() >= 1024 * 1024:  # Process every 1MB
                self._process_buffer()

    def _process_buffer(self) -> None:
        if self.is_processing:
            return

        self.is_processing = True
        buffer_data = self.buffer.getvalue()
        self.buffer = io.BytesIO()

        try:
            # Upload audio to AssemblyAI
            audio_file = self.client.upload(buffer_data)

            # Start transcription
            transcript = self.client.transcribe(
                audio_file,
                config=aai.TranscriptionConfig(language_code="en")
            )

            # Analyze transcription
            analysis = self.analyzer.analyze(transcript.text, {
                'type': 'live_stream',
                'timestamp': datetime.now().isoformat()
            })

            # Send results through websocket
            self.websocket.send(json.dumps({
                'type': 'transcription',
                'text': transcript.text,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }))

        except Exception as e:
            self.websocket.send(json.dumps({
                'type': 'error',
                'error': 'Stream processing failed',
                'details': str(e)
            }))
        finally:
            self.is_processing = False
