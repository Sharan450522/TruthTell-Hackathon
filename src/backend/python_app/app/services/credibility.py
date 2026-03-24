import re
from typing import Dict, List, Any
from config import Config

class CredibilityAnalyzer:
    def __init__(self, cache_manager):
        self.patterns = {}
        self.cache_manager = cache_manager
        self.compile_patterns()

    def compile_patterns(self):
        for category, keywords in Config.CREDIBILITY_RULES['SUSPICIOUS_KEYWORDS'].items():
            for item in keywords:
                self.patterns[item['word']] = {
                    'regex': re.compile(item['word'], re.IGNORECASE),
                    'weight': item['weight'],
                    'category': category
                }

        for indicator in Config.CREDIBILITY_RULES['CREDIBLE_INDICATORS']:
            self.patterns[indicator['phrase']] = {
                'regex': re.compile(indicator['phrase'], re.IGNORECASE),
                'weight': indicator['weight'],
                'category': 'CREDIBLE'
            }

    def analyze(self, text: str, metadata: Dict = None) -> Dict[str, Any]:
        if not metadata:
            metadata = {}

        cache_key = text[:100]
        cached = self.cache_manager.get('analysis', cache_key)
        if cached:
            return cached

        score = 100
        detected_patterns = []
        analysis_details = {
            'sensationalism': 0,
            'clickbait': 0,
            'conspiracy': 0,
            'credible_indicators': 0
        }

        for pattern, info in self.patterns.items():
            matches = len(info['regex'].findall(text))
            if matches > 0:
                score += info['weight'] * matches
                analysis_details[info['category'].lower()] += matches
                detected_patterns.append({
                    'pattern': pattern,
                    'category': info['category'],
                    'matches': matches,
                    'impact': info['weight'] * matches
                })

        score = max(10, min(score, 100))

        result = {
            'verdict': 'Highly Credible' if score > 70 else 'Somewhat Credible' if score > 50 else 'Low Credibility',
            'confidence': round(score),
            'analysis_details': analysis_details,
            'detected_patterns': detected_patterns,
            'metadata': {
                **metadata,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

        self.cache_manager.set('analysis', cache_key, result)
        return result
