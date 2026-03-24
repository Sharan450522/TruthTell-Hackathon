from typing import Dict, Any, Optional, List
import asyncio
import json
import logging
from datetime import datetime
import assemblyai as aai

logger = logging.getLogger(__name__)

class SpeechProcessor:
    def __init__(self, api_key: str, credibility_analyzer: Any):
        self.client = aai.Client(api_key)
        self.credibility_analyzer = credibility_analyzer
        self.active_transcriptions: Dict[str, Any] = {}

    async def process_speech(self, audio_data: bytes, session_id: str) -> Optional[Dict[str, Any]]:
        """Process speech from audio data and return transcription with analysis."""
        try:
            # Upload audio to AssemblyAI
            upload_response = self.client.upload(audio_data)

            # Start transcription
            transcript = self.client.transcribe(
                upload_response,
                config=aai.TranscriptionConfig(
                    language_code="en",
                    punctuate=True,
                    format_text=True
                )
            )

            # Store transcription reference
            self.active_transcriptions[session_id] = {
                'transcript_id': transcript.id,
                'start_time': datetime.now()
            }

            # Wait for completion
            while transcript.status != 'completed':
                await asyncio.sleep(1)
                transcript = self.client.get_transcription(transcript.id)

            # Analyze transcription
            analysis = self.credibility_analyzer.analyze(
                transcript.text,
                {'type': 'speech', 'session_id': session_id}
            )

            return {
                'text': transcript.text,
                'analysis': analysis,
                'words': transcript.words,
                'confidence': transcript.confidence
            }

        except Exception as e:
            logger.error(f'Speech processing error: {str(e)}')
            return None

    def get_transcription_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active transcription."""
        if session_id in self.active_transcriptions:
            trans_data = self.active_transcriptions[session_id]
            try:
                transcript = self.client.get_transcription(trans_data['transcript_id'])
                return {
                    'status': transcript.status,
                    'duration': (datetime.now() - trans_data['start_time']).seconds
                }
            except Exception as e:
                logger.error(f'Status check error: {str(e)}')
        return None
