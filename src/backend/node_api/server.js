require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const WebSocket = require('ws');
const { AssemblyAI } = require('assemblyai');
const { spawn } = require('child_process');
const stream = require('stream');
const path = require('path');
const cors = require('cors');
const axios = require('axios');
const feedparser = require('feedparser-promised');
const winston = require('winston');
const stringSimilarity = require('string-similarity');
const cheerio = require('cheerio');

// Constants and Configurations
const PORT = process.env.PORT || 3000;
const NEWS_UPDATE_INTERVAL = 300000; // 5 minutes
const CACHE_CLEANUP_INTERVAL = 3600000; // 1 hour

// Initialize Express and WebSocket server
const app = express();
const server = require('http').createServer(app);
const wss = new WebSocket.Server({ server });

// Configure logging with Winston
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'error.log', level: 'error' }),
        new winston.transports.File({ filename: 'combined.log' })
    ]
});

// Credibility Rules Configuration
const CREDIBILITY_RULES = {
    SUSPICIOUS_KEYWORDS: {
        SENSATIONALISM: [
            { word: 'shocking', weight: -8 },
            { word: 'unbelievable', weight: -7 },
            { word: 'sensational', weight: -6 },
            { word: 'breaking', weight: -5 },
            { word: 'exclusive', weight: -5 }
        ],
        CLICKBAIT: [
            { word: 'you won\'t believe', weight: -8 },
            { word: 'mind-blowing', weight: -7 },
            { word: 'viral', weight: -6 },
            { word: 'secret', weight: -5 }
        ],
        CONSPIRACY: [
            { word: 'conspiracy', weight: -8 },
            { word: 'exposed', weight: -7 },
            { word: 'they don\'t want you to know', weight: -8 },
            { word: 'hidden truth', weight: -7 }
        ]
    },
    CREDIBLE_INDICATORS: [
        { phrase: 'according to research', weight: 5 },
        { phrase: 'studies show', weight: 4 },
        { phrase: 'experts say', weight: 3 },
        { phrase: 'official statement', weight: 5 }
    ]
};

// News Sources Configuration
const NEWS_SOURCES = {
    RSS_FEEDS: [
        {
            url: 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms',
            name: 'Times of India',
            reliability: 0.8
        },
        {
            url: 'https://www.thehindu.com/news/national/feeder/default.rss',
            name: 'The Hindu',
            reliability: 0.85
        }
    ],
    GNEWS: {
        endpoint: 'https://gnews.io/api/v4/top-headlines',
        params: {
            country: 'in',
            lang: 'en',
            max: 10,
            token: process.env.GNEWS_API_KEY
        }
    }
};

// Optimized Cache System
class CacheManager {
    constructor() {
        this.caches = new Map();
        this.initializeCaches();
    }

    initializeCaches() {
        ['audio', 'news', 'analysis', 'liveStreams'].forEach(cacheType => {
            this.caches.set(cacheType, new Map());
        });
    }

    get(type, key) {
        const cache = this.caches.get(type);
        const item = cache.get(key);
        if (item && !this.isExpired(item.timestamp)) {
            return item.data;
        }
        return null;
    }

    set(type, key, data, ttl = 3600000) {
        const cache = this.caches.get(type);
        cache.set(key, {
            data,
            timestamp: Date.now(),
            ttl
        });
    }

    delete(type, key) {
        const cache = this.caches.get(type);
        cache.delete(key);
    }

    clear(type) {
        if (type) {
            this.caches.get(type).clear();
        } else {
            this.caches.forEach(cache => cache.clear());
        }
    }

    isExpired(timestamp, ttl = 3600000) {
        return (Date.now() - timestamp) > ttl;
    }

    cleanup() {
        this.caches.forEach((cache, type) => {
            for (const [key, value] of cache.entries()) {
                if (this.isExpired(value.timestamp, value.ttl)) {
                    this.delete(type, key);
                }
            }
        });
    }
}

// Initialize Cache
const cacheManager = new CacheManager();

// Optimized Buffer Management
class OptimizedSlidingBuffer {
    constructor(maxDuration = 60000, chunkSize = 10000) {
        this.maxDuration = maxDuration;
        this.chunkSize = chunkSize;
        this.chunks = [];
        this.totalDuration = 0;
        this.lastProcessedTimestamp = 0;
    }

    addChunk(chunk, duration) {
        const now = Date.now();
        this.chunks.push({ 
            chunk, 
            duration,
            timestamp: now 
        });
        this.totalDuration += duration;

        while (this.totalDuration > this.maxDuration) {
            const removed = this.chunks.shift();
            this.totalDuration -= removed.duration;
        }

        return this.shouldProcess(now);
    }

    shouldProcess(now) {
        return (now - this.lastProcessedTimestamp) >= this.chunkSize;
    }

    getBuffer() {
        this.lastProcessedTimestamp = Date.now();
        return Buffer.concat(this.chunks.map(c => c.chunk));
    }

    clear() {
        this.chunks = [];
        this.totalDuration = 0;
    }
}

// Optimized Credibility Analyzer
class CredibilityAnalyzer {
    constructor() {
        this.patterns = new Map();
        this.compilePatterns();
    }

    compilePatterns() {
        Object.entries(CREDIBILITY_RULES.SUSPICIOUS_KEYWORDS).forEach(([category, keywords]) => {
            keywords.forEach(({ word, weight }) => {
                this.patterns.set(word, {
                    regex: new RegExp(word, 'gi'),
                    weight,
                    category
                });
            });
        });

        CREDIBILITY_RULES.CREDIBLE_INDICATORS.forEach(({ phrase, weight }) => {
            this.patterns.set(phrase, {
                regex: new RegExp(phrase, 'gi'),
                weight,
                category: 'CREDIBLE'
            });
        });
    }

    analyze(text, metadata = {}) {
        const cacheKey = text.substring(0, 100);
        const cached = cacheManager.get('analysis', cacheKey);
        if (cached) return cached;

        let score = 100;
        const detected_patterns = [];
        const analysis_details = {
            sensationalism: 0,
            clickbait: 0,
            conspiracy: 0,
            credible_indicators: 0
        };

        for (const [pattern, { regex, weight, category }] of this.patterns) {
            const matches = (text.match(regex) || []).length;
            if (matches > 0) {
                score += weight * matches;
                analysis_details[category.toLowerCase()] += matches;
                detected_patterns.push({
                    pattern,
                    category,
                    matches,
                    impact: weight * matches
                });
            }
        }

        score = Math.max(10, Math.min(score, 100));

        const result = {
            verdict: score > 70 ? 'Highly Credible' : score > 50 ? 'Somewhat Credible' : 'Low Credibility',
            confidence: Math.round(score),
            analysis_details,
            detected_patterns,
            metadata: {
                ...metadata,
                analysis_timestamp: new Date().toISOString()
            }
        };

        cacheManager.set('analysis', cacheKey, result);
        return result;
    }
}

const SourceVerifier = {
    // Reliability scores for different source types
    RELIABILITY_SCORES: {
        ACADEMIC: 0.9,
        GOVERNMENT: 0.85,
        NEWS_AGENCY: 0.75,
        FACT_CHECK_ORG: 0.8,
        SOCIAL_MEDIA: 0.3,
        UNKNOWN: 0.4
    },

    // Domain patterns for source classification
    DOMAIN_PATTERNS: {
        ACADEMIC: [/\.edu$/, /\.ac\.[a-z]{2}$/],
        GOVERNMENT: [/\.gov$/, /\.gov\.[a-z]{2}$/],
        NEWS_AGENCY: [
            'reuters.com',
            'apnews.com',
            'bloomberg.com',
            'afp.com'
        ],
        FACT_CHECK_ORG: [
            'snopes.com',
            'factcheck.org',
            'politifact.com'
        ]
    },

    async verifySource(url) {
        try {
            const domain = new URL(url).hostname;
            const sourceType = this.classifySource(domain);
            const baseScore = this.RELIABILITY_SCORES[sourceType];

            // Additional verification checks
            const [sslValid, hasHTTPS] = await Promise.all([
                this.verifySSL(url),
                url.startsWith('https')
            ]);

            // Cross-reference with fact-checking databases
            const factCheckData = await this.crossReferenceFactCheckers(url);

            return {
                score: this.calculateFinalScore(baseScore, {
                    sslValid,
                    hasHTTPS,
                    factCheckData
                }),
                sourceType,
                verificationDetails: {
                    domain,
                    sslValid,
                    hasHTTPS,
                    factCheckData
                }
            };
        } catch (error) {
            console.error('Source verification error:', error);
            return {
                score: this.RELIABILITY_SCORES.UNKNOWN,
                sourceType: 'UNKNOWN',
                error: error.message
            };
        }
    },

    classifySource(domain) {
        for (const [type, patterns] of Object.entries(this.DOMAIN_PATTERNS)) {
            if (patterns.some(pattern => 
                pattern instanceof RegExp ? 
                pattern.test(domain) : 
                domain.includes(pattern)
            )) {
                return type;
            }
        }
        return 'UNKNOWN';
    },

    async verifySSL(url) {
        try {
            const response = await fetch(url, {
                method: 'HEAD',
                mode: 'no-cors'
            });
            return response.ok;
        } catch {
            return false;
        }
    },

    async crossReferenceFactCheckers(url) {
        const factCheckAPIs = [
            `https://factchecktools.googleapis.com/v1alpha1/claims:search?key=${process.env.GOOGLE_FACT_CHECK_API_KEY}&query=${encodeURIComponent(url)}`,
            // Add other fact-checking APIs as needed
        ];

        const results = await Promise.allSettled(
            factCheckAPIs.map(api => fetch(api).then(res => res.json()))
        );

        return results
            .filter(result => result.status === 'fulfilled')
            .map(result => result.value)
            .flat();
    },

    calculateFinalScore(baseScore, factors) {
        let score = baseScore;
        
        if (!factors.sslValid) score *= 0.9;
        if (!factors.hasHTTPS) score *= 0.9;
        
        if (factors.factCheckData?.length > 0) {
            const factCheckImpact = factors.factCheckData.reduce((acc, check) => {
                return acc + (check.rating === 'true' ? 0.1 : -0.1);
            }, 0);
            score = Math.max(0.1, Math.min(1, score + factCheckImpact));
        }

        return Math.round(score * 100);
    }
};

// Integration with existing CredibilityAnalyzer
class EnhancedCredibilityAnalyzer extends CredibilityAnalyzer {
    async analyze(text, metadata = {}) {
        const baseAnalysis = super.analyze(text, metadata);
        
        if (metadata.url) {
            const sourceVerification = await SourceVerifier.verifySource(metadata.url);
            const combinedScore = Math.round(
                (baseAnalysis.confidence + sourceVerification.score) / 2
            );

            return {
                ...baseAnalysis,
                confidence: combinedScore,
                source_verification: sourceVerification,
                verdict: this.getEnhancedVerdict(combinedScore, sourceVerification)
            };
        }

        return baseAnalysis;
    }

    getEnhancedVerdict(score, sourceVerification) {
        if (score > 80 && sourceVerification.sourceType !== 'UNKNOWN') {
            return 'Highly Reliable';
        } else if (score > 60) {
            return 'Moderately Reliable';
        } else {
            return 'Low Reliability';
        }
    }
}



class AdvancedTextAnalyzer extends EnhancedCredibilityAnalyzer {
    constructor() {
        super();
        this.statementPatterns = {
            factual: /(?:according to|studies show|research indicates|data shows|evidence suggests|experts say|statistics indicate)/gi,
            claims: /(?:claims|suggests|believes|argues|states|asserts|proposes)/gi,
            citations: /(?:\([^)]*\d{4}[^)]*\)|\[\d+\])/g
        };
    }

    async analyzeArticleText(text, metadata = {}) {
        const baseAnalysis = await super.analyze(text, metadata);
        
        // Extract and analyze statements
        const statements = this.extractStatements(text);
        
        // Analyze citations and references
        const citationAnalysis = this.analyzeCitations(text);
        
        // Analyze writing style and tone
        const styleAnalysis = this.analyzeStyle(text);
        
        // Check for balanced reporting
        const balanceAnalysis = this.analyzeBalance(text);

        return {
            ...baseAnalysis,
            detailed_analysis: {
                statements,
                citations: citationAnalysis,
                style: styleAnalysis,
                balance: balanceAnalysis
            }
        };
    }

    extractStatements(text) {
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
        
        return {
            factual: sentences.filter(s => this.statementPatterns.factual.test(s)),
            claims: sentences.filter(s => this.statementPatterns.claims.test(s)),
            total_statements: sentences.length
        };
    }

    analyzeCitations(text) {
        const citations = text.match(this.statementPatterns.citations) || [];
        
        return {
            count: citations.length,
            density: citations.length / (text.length / 1000), // Citations per 1000 characters
            has_references: citations.length > 0
        };
    }

    analyzeStyle(text) {
        const words = text.split(/\s+/);
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
        
        return {
            avg_sentence_length: words.length / sentences.length,
            academic_style: this.checkAcademicStyle(text),
            emotional_language: this.checkEmotionalLanguage(text)
        };
    }

    analyzeBalance(text) {
        const positivePatterns = /(?:benefit|advantage|improvement|success|positive)/gi;
        const negativePatterns = /(?:drawback|limitation|challenge|negative|failure)/gi;
        
        const positiveMatches = (text.match(positivePatterns) || []).length;
        const negativeMatches = (text.match(negativePatterns) || []).length;
        
        return {
            positive_mentions: positiveMatches,
            negative_mentions: negativeMatches,
            balance_score: Math.abs(0.5 - (positiveMatches / (positiveMatches + negativeMatches || 1)))
        };
    }

    checkAcademicStyle(text) {
        const academicPhrases = /(?:furthermore|moreover|however|consequently|therefore|thus|hence|accordingly)/gi;
        return (text.match(academicPhrases) || []).length;
    }

    checkEmotionalLanguage(text) {
        const emotionalWords = /(?:amazing|terrible|wonderful|horrible|awesome|devastating|incredible|outrageous)/gi;
        return (text.match(emotionalWords) || []).length;
    }
}



// Initialize AssemblyAI
const assemblyAI = new AssemblyAI({
    apiKey: process.env.ASSEMBLYAI_API_KEY
});

// Optimized Stream Processor
class StreamProcessor {
    constructor(assemblyAI, ws) {
        this.assemblyAI = assemblyAI;
        this.ws = ws;
        this.buffer = new OptimizedSlidingBuffer();
        this.analyzer = new CredibilityAnalyzer();
        this.transcriptionQueue = [];
        this.isProcessing = false;
    }

    async processChunk(chunk, duration) {
        if (!this.buffer.addChunk(chunk, duration)) {
            return;
        }

        this.transcriptionQueue.push({
            buffer: this.buffer.getBuffer(),
            timestamp: Date.now()
        });

        if (!this.isProcessing) {
            this.processQueue();
        }
    }

    async processQueue() {
        if (this.isProcessing || this.transcriptionQueue.length === 0) {
            return;
        }

        this.isProcessing = true;
        const { buffer, timestamp } = this.transcriptionQueue.shift();

        try {
            const audioStream = new stream.PassThrough();
            audioStream.end(buffer);

            const [audioFile] = await Promise.all([
                this.assemblyAI.files.upload(audioStream, {
                    fileName: `chunk-${timestamp}.mp3`,
                    contentType: 'audio/mp3'
                })
            ]);

            const transcript = await this.assemblyAI.transcripts.transcribe({
                audio: audioFile,
                language_code: 'en',
                word_boost: Array.from(this.analyzer.patterns.keys()),
                boost_param: "high"
            });

            const analysis = this.analyzer.analyze(transcript.text, {
                type: 'live_stream',
                timestamp
            });

            this.ws.send(JSON.stringify({
                type: 'transcription',
                text: transcript.text,
                analysis,
                timestamp
            }));

        } catch (error) {
            logger.error('Stream processing error:', error);
            this.ws.send(JSON.stringify({
                type: 'error',
                error: 'Stream processing failed',
                details: error.message
            }));
        } finally {
            this.isProcessing = false;
            if (this.transcriptionQueue.length > 0) {
                this.processQueue();
            }
        }
    }
}

// Utility Functions
const detectPlatform = (url) => {
    try {
        const urlObj = new URL(url);
        const domain = urlObj.hostname.toLowerCase();
        
        const platformMap = {
            'youtube.com': 'youtube',
            'youtu.be': 'youtube',
            'instagram.com': 'instagram',
            'facebook.com': 'facebook',
            'fb.com': 'facebook',
            'tiktok.com': 'tiktok',
            'twitter.com': 'twitter',
            'vimeo.com': 'vimeo'
        };

        for (const [key, value] of Object.entries(platformMap)) {
            if (domain.includes(key)) return value;
        }
        
        return 'unknown';
    } catch (error) {
        throw new Error('Invalid URL format');
    }
};

// News Fetching Functions
const fetchGNewsArticles = async () => {
    try {
        if (!process.env.GNEWS_API_KEY) {
            logger.error('GNews API key is not configured');
            return [];
        }

        const response = await axios.get(NEWS_SOURCES.GNEWS.endpoint, {
            params: NEWS_SOURCES.GNEWS.params
        });

        if (response.data?.articles) {
            return response.data.articles.map(article => ({
                title: article.title,
                description: article.description,
                url: article.url,
                source: article.source.name,
                published: article.publishedAt
            }));
        }
        return [];
    } catch (error) {
        logger.error(`GNews API Error: ${error.message}`);
        return [];
    }
};

const fetchTrendingNews = async () => {
    const cachedNews = cacheManager.get('news', 'trending');
    if (cachedNews) return cachedNews;

    try {
        const [rssResults, gnewsResults] = await Promise.all([
            Promise.all(NEWS_SOURCES.RSS_FEEDS.map(async (source) => {
                try {
                    const items = await feedparser.parse(source.url);
                    return items.slice(0, 10).map(item => ({
                        title: item.title,
                        description: item.description || item.summary,
                        url: item.link,
                        source: source.name,
                        reliability: source.reliability,
                        published: item.pubDate
                    }));
                } catch (error) {
                    logger.error(`RSS feed error for ${source.name}:`, error);
                    return [];
                }
            })).then(results => results.flat()),
            fetchGNewsArticles()
        ]);

        const analyzer = new CredibilityAnalyzer();
        const allNews = [...rssResults, ...gnewsResults];

        const uniqueNews = allNews.reduce((acc, current) => {
            const isDuplicate = acc.some(item => 
                stringSimilarity.compareTwoStrings(item.title, current.title) > 0.8
            );
            if (!isDuplicate) {
                const analysis = analyzer.analyze(
                    `${current.title} ${current.description || ''}`,
                    { source: current.source, published: current.published }
                );
                acc.push({ ...current, analysis });
            }
            return acc;
        }, []);

        const sortedNews = uniqueNews.sort((a, b) => {
            const timeWeight = 0.3;
            const credibilityWeight = 0.7;
            
            const timeA = new Date(a.published).getTime();
            const timeB = new Date(b.published).getTime();
            
            const timeScore = (timeA - timeB) * timeWeight;
            const credScore = (a.analysis.confidence - b.analysis.confidence) * credibilityWeight;
            
            return (credScore + timeScore) * -1;
        });

        const result = sortedNews.slice(0, 15);
        cacheManager.set('news', 'trending', result, NEWS_UPDATE_INTERVAL);
        return result;
    } catch (error) {
        logger.error('Error in fetchTrendingNews:', error);
        return [];
    }
};



// Express Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// API Routes


async function scrapeArticle(url) {
    try {
        const response = await axios.get(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        });
        
        const $ = cheerio.load(response.data);
        
        // Remove unwanted elements
        $('script, style, nav, header, footer, .ads, #comments').remove();
        
        // Extract article content
        const article = $('article').length ? 
            $('article') : 
            $('main').length ? 
                $('main') : 
                $('body');
                
        // Get text from paragraphs
        const paragraphs = [];
        article.find('p').each((_, element) => {
            const text = $(element).text().trim();
            if (text.length > 30) { // Filter out short snippets
                paragraphs.push(text);
            }
        });
        
        return paragraphs.join(' ');
    } catch (error) {
        console.error("Error scraping article:", error);
        return null;
    }
}
// API Routes (continued)


app.post('/api/analyze-article', async (req, res) => {
    const { url, language = 'en' } = req.body;

    if (!url) {
        return res.status(400).json({ error: 'Article URL is required' });
    }

    try {
        const articleText = await scrapeArticle(url);
        
        if (!articleText) {
            throw new Error('Failed to extract article content');
        }

        const analyzer = new AdvancedTextAnalyzer();
        const analysis = await analyzer.analyzeArticleText(articleText, {
            type: 'article',
            url,
            language
        });

        res.json({
            text: articleText,
            analysis,
            success: true
        });
    } catch (error) {
        logger.error('Article analysis error:', error);
        res.status(500).json({
            error: 'Failed to analyze article',
            details: error.message
        });
    }
});

app.post('/api/analyze-text', async (req, res) => {
    const { text, language = 'en' } = req.body;

    if (!text) {
        return res.status(400).json({ error: 'Text is required' });
    }

    try {
        const analyzer = new CredibilityAnalyzer();
        const analysis = analyzer.analyze(text, {
            type: 'manual_input',
            language
        });

        res.json({
            text,
            analysis,
            success: true
        });
    } catch (error) {
        logger.error('Text analysis error:', error);
        res.status(500).json({
            error: 'Failed to analyze text',
            details: error.message
        });
    }
});

app.post('/api/transcribe-recorded', async (req, res) => {
    const { video_url, language = 'en' } = req.body;
    let ytDlpProcess = null;

    if (!video_url) {
        return res.status(400).json({ error: 'Video URL is required' });
    }

    try {
        const platform = detectPlatform(video_url);
        logger.info(`Starting transcription for ${platform} video: ${video_url}`);
        
        ytDlpProcess = spawn('yt-dlp', [
            '-x',
            '--audio-format', 'mp3',
            '--output', '-',
            '--no-playlist',
            video_url
        ]);

        const audioChunks = [];
        
        ytDlpProcess.stdout.on('data', chunk => {
            audioChunks.push(chunk);
        });

        ytDlpProcess.stderr.on('data', data => {
            logger.info(`Download progress: ${data}`);
        });

        const processVideo = new Promise((resolve, reject) => {
            ytDlpProcess.on('close', code => {
                if (code === 0 && audioChunks.length > 0) {
                    resolve(Buffer.concat(audioChunks));
                } else {
                    reject(new Error(`Download failed with code ${code}`));
                }
            });
            ytDlpProcess.on('error', reject);
        });

        const audioBuffer = await processVideo;
        
        if (!audioBuffer || audioBuffer.length === 0) {
            throw new Error('No audio data received from video');
        }

        const audioStream = new stream.PassThrough();
        audioStream.end(audioBuffer);

        const audioFile = await assemblyAI.files.upload(audioStream, {
            fileName: `${platform}-${Date.now()}.mp3`,
            contentType: 'audio/mp3'
        });

        const transcript = await assemblyAI.transcripts.transcribe({
            audio: audioFile,
            language_code: language
        });

        const analyzer = new CredibilityAnalyzer();
        const analysis = analyzer.analyze(transcript.text, {
            type: 'recorded_video',
            platform,
            url: video_url
        });

        res.json({
            text: transcript.text,
            platform,
            analysis,
            success: true
        });
    } catch (error) {
        logger.error('Transcription Error:', error);
        res.status(500).json({
            error: 'Failed to transcribe video',
            details: error.message,
            platform: detectPlatform(video_url)
        });
    } finally {
        if (ytDlpProcess && !ytDlpProcess.killed) {
            try {
                ytDlpProcess.kill();
            } catch (err) {
                logger.error('Error killing yt-dlp process:', err);
            }
        }
    }
});

app.get('/api/news-stream', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const sendNews = async () => {
        try {
            const news = await fetchTrendingNews();
            res.write(`data: ${JSON.stringify(news)}\n\n`);
        } catch (error) {
            logger.error('News stream error:', error);
        }
    };

    const newsInterval = setInterval(sendNews, NEWS_UPDATE_INTERVAL);
    sendNews();

    req.on('close', () => {
        clearInterval(newsInterval);
    });
});

app.get('/api/trending-news', async (req, res) => {
    try {
        const news = await fetchTrendingNews();
        res.json(news);
    } catch (error) {
        logger.error('Error fetching trending news:', error);
        res.status(500).json({
            error: 'Failed to fetch trending news',
            details: error.message
        });
    }
});

app.get('/api/news-sources', (req, res) => {
    const sources = NEWS_SOURCES.RSS_FEEDS.map(source => ({
        name: source.name,
        url: source.url
    }));
    sources.push({ name: 'GNews', url: 'https://gnews.io/' });
    
    res.json(sources);
});

app.post('/api/clear-cache', (req, res) => {
    cacheManager.cleanup();
    res.json({ message: 'Cache cleared successfully' });
});

app.get('/health', (req, res) => {
    res.json({ 
        status: 'ok',
        timestamp: new Date().toISOString(),
        cache_stats: {
            audio: cacheManager.caches.get('audio').size,
            news: cacheManager.caches.get('news').size,
            analysis: cacheManager.caches.get('analysis').size
        }
    });
});

// WebSocket Handler for Live Streaming
wss.on('connection', (ws) => {
    logger.info('New WebSocket connection established');
    let currentStream = null;

    ws.on('message', async (message) => {
        try {
            const data = JSON.parse(message);

            if (data.type === 'start_live') {
                const platform = detectPlatform(data.url);
                logger.info(`Starting live stream processing for ${platform}:`, data.url);

                if (currentStream) {
                    currentStream.kill();
                }

                const ytDlpProcess = spawn('yt-dlp', [
                    '-x',
                    '--audio-format', 'mp3',
                    '--output', '-',
                    '--no-playlist',
                    '--live-from-start',
                    data.url
                ]);

                const streamProcessor = new StreamProcessor(assemblyAI, ws);

                ytDlpProcess.stdout.on('data', chunk => {
                    streamProcessor.processChunk(chunk, 1);
                });

                ytDlpProcess.stderr.on('data', data => {
                    logger.info(`Stream progress: ${data}`);
                });

                currentStream = ytDlpProcess;
                cacheManager.set('liveStreams', `${platform}-${Date.now()}`, currentStream);

                ws.send(JSON.stringify({
                    type: 'status',
                    message: `Live stream processing started for ${platform}`,
                    platform
                }));
            }
        } catch (error) {
            logger.error('WebSocket message error:', error);
            ws.send(JSON.stringify({
                type: 'error',
                error: 'Failed to process message',
                details: error.message
            }));
        }
    });

    ws.on('close', () => {
        logger.info('WebSocket connection closed');
        if (currentStream) {
            currentStream.kill();
        }
    });

    ws.send(JSON.stringify({
        type: 'status',
        message: 'Connected to transcription service'
    }));
});

// Error Handler
app.use((err, req, res, next) => {
    logger.error('Server Error:', err);
    res.status(500).json({
        error: 'Internal Server Error',
        details: err.message
    });
});

// Periodic Cache Cleanup
setInterval(() => {
    cacheManager.cleanup();
}, CACHE_CLEANUP_INTERVAL);

// Graceful Shutdown
const gracefulShutdown = () => {
    logger.info('Initiating graceful shutdown...');
    
    cacheManager.caches.get('liveStreams').forEach(stream => stream.kill());
    cacheManager.clear();
    
    server.close(() => {
        logger.info('Server shut down complete');
        process.exit(0);
    });
};

process.on('SIGTERM', gracefulShutdown);
process.on('SIGINT', gracefulShutdown);

// Start Server
server.listen(PORT, () => {
    logger.info(`Server running at http://localhost:${PORT}`);
});

module.exports = { app, server };