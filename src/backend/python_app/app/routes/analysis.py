from flask import Blueprint, request, jsonify
from app.services.credibility import CredibilityAnalyzer
from app.services.cache import CacheManager
import logging

bp = Blueprint('analysis', __name__)
logger = logging.getLogger(__name__)

@bp.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text')
    language = data.get('language', 'en')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        analysis = CredibilityAnalyzer(CacheManager()).analyze(text, {
            'type': 'manual_input',
            'language': language
        })

        return jsonify({
            'text': text,
            'analysis': analysis,
            'success': True
        })
    except Exception as e:
        logger.error(f'Text analysis error: {str(e)}')
        return jsonify({
            'error': 'Failed to analyze text',
            'details': str(e)
        }), 500
