# Truth Tell Fake News Pipeline

This document describes the current end-to-end pipeline for article URLs, YouTube recorded videos, audio files, video files, and live streams.

## Common Flow

```text
Input
-> detect source type
-> extract text
-> normalize and merge useful text
-> save extracted content in SQLite
-> store text chunks in Pinecone
-> run fake-news classifier
-> save prediction in SQLite
-> return fake / real / uncertain with confidence
-> optional human review label
```

Pinecone uses integrated embedding:

```text
Index: truth-tell-news
Model: llama-text-embed-v2
Vector type: dense
Dimension: 1024
Metric: cosine
Field map: text
Namespace: __default__
```

No OpenAI API key is required when `PINECONE_EMBEDDING_PROVIDER=pinecone`.

## 1. Article URL

```text
Article URL
-> scrape webpage with requests + BeautifulSoup
-> remove script/style/nav/header/footer/ads/comments
-> extract paragraph text from article/main/body
-> create content piece: article_text
-> normalize text
-> aggregate article body as classifier input
-> save source in SQLite
-> save extracted_content rows in SQLite
-> chunk article text
-> upsert chunks to Pinecone
-> run pipeline.pkl transform
-> run ensemble_model.pkl predict/predict_proba
-> map prediction to fake/real/uncertain
-> save prediction in SQLite
-> return result to UI
```

API route:

```text
POST /api/analyze-article
```

Request body:

```json
{
  "url": "https://example.com/news-story"
}
```

Response includes:

```json
{
  "text": "extracted article text",
  "classification_text": "text sent to model",
  "analysis": {
    "label": "fake",
    "confidence": 0.73,
    "status": "likely_fake",
    "source_id": "...",
    "prediction_id": "...",
    "vector_storage": {
      "embedding_provider": "pinecone",
      "upserted": 1
    }
  },
  "success": true
}
```

## 2. YouTube Recorded Video

```text
YouTube URL
-> yt-dlp downloads video to temp/downloaded_video.mp4
-> VideoProcessor processes local video
-> extract frames
-> extract audio using moviepy/ffmpeg
-> transcribe audio using Whisper
-> OCR selected frames using Tesseract
-> optionally generate visual features using CLIP
-> optionally generate GPT-2 local inference summary
-> create content pieces:
   transcript
   frame_ocr
-> normalize and merge transcript + OCR
-> save source in SQLite
-> save extracted transcript/OCR chunks in SQLite
-> upsert chunks to Pinecone
-> run fake-news classifier
-> save prediction in SQLite
-> return transcript, OCR, frame previews, and classifier result
-> delete temp video file
```

API route:

```text
POST /api/analyze-recorded-video
```

Request body:

```json
{
  "file_path": "https://youtu.be/VIDEO_ID"
}
```

Classification input priority:

```text
transcript > OCR
```

## 3. Audio File

```text
Uploaded audio file
-> save file temporarily in temp/
-> detect audio extension
-> transcribe audio using Whisper
-> create content piece: transcript
-> normalize transcript
-> save source in SQLite
-> save transcript chunks in SQLite
-> upsert transcript chunks to Pinecone
-> run fake-news classifier
-> save prediction in SQLite
-> return transcript + classifier result
-> delete temp file
```

API route:

```text
POST /api/analyze-media
```

Form-data:

```text
media = audio file
```

Supported audio extensions:

```text
.mp3
.wav
.m4a
.aac
```

## 4. Video File

```text
Uploaded video file
-> save file temporarily in temp/
-> detect video extension
-> VideoProcessor processes local video
-> extract frames
-> extract audio using moviepy/ffmpeg
-> transcribe audio using Whisper
-> OCR frames using Tesseract
-> extract CLIP visual features
-> generate optional GPT-2 inference summary
-> create content pieces:
   transcript
   frame_ocr
-> normalize and merge transcript + OCR
-> save source in SQLite
-> save extracted chunks in SQLite
-> upsert transcript/OCR chunks to Pinecone
-> classify merged transcript + OCR
-> save prediction in SQLite
-> return transcript, OCR, thumbnails, and classifier result
-> delete temp file
```

API route:

```text
POST /api/analyze-media
```

Form-data:

```text
media = video file
```

Supported video extensions:

```text
.mp4
.mov
.avi
```

There is also a direct video route:

```text
POST /api/analyze-video
```

The UI mainly uses `/api/analyze-media`.

## 5. Live Stream

```text
Live stream URL
-> Socket.IO start_live event
-> loop every 30 seconds
-> capture short live audio segment with yt-dlp
-> preferred format: ba/bestaudio/best
-> fallback: resolve audio stream URL with yt-dlp
-> capture audio with ffmpeg
-> optional: try to capture video preview frame
-> upload audio segment to AssemblyAI
-> get transcript
-> create content piece: transcript
-> save source in SQLite
-> save transcript chunk in SQLite
-> upsert transcript chunk to Pinecone
-> classify segment transcript
-> mark needs_verification = true
-> smooth confidence across segments
-> emit live result to UI over Socket.IO
-> repeat until user stops live stream
```

Socket events:

```text
start_live
stop_live
transcription
video_fragment
```

Example live response:

```json
{
  "text": "segment transcript",
  "analysis": {
    "label": "fake",
    "confidence": 0.62,
    "status": "likely_fake",
    "needs_verification": true,
    "cumulative_confidence": 0.59
  },
  "type": "Segment"
}
```

For live streams:

```text
needs_verification = true
```

Live content can be incomplete, changing, or hard to verify from a classifier alone.

## Classifier Logic

All analysis paths eventually call:

```text
pipeline.pkl
ensemble_model.pkl
```

Runtime:

```text
merged_text
-> pipeline.transform([merged_text])
-> ensemble_model.predict(...)
-> ensemble_model.predict_proba(...)
```

Class mapping:

```text
1 = real
0 = fake
```

Confidence:

```text
confidence = max(class probabilities)
```

Decision thresholds:

```text
confidence < 0.55
-> label = uncertain
-> status = uncertain

0.55 <= confidence < 0.80
-> label = fake/real
-> status = likely_fake / likely_real

confidence >= 0.80
-> label = fake/real
-> status = high_confidence
```

## Storage

SQLite stores structured records:

```text
data/truth_tell.db
```

Tables:

```text
sources
extracted_content
predictions
review_labels
```

Pinecone stores searchable text chunks:

```text
Index: truth-tell-news
Embedding provider: Pinecone
Model: llama-text-embed-v2
Namespace: __default__
Field map: text
```

Stored chunk metadata:

```json
{
  "source_id": "...",
  "source_type": "recorded_video",
  "content_type": "transcript",
  "chunk_index": 0,
  "prediction_id": "..."
}
```

## Human Feedback

After prediction, the UI can save a human review label:

```text
POST /api/review-label
```

Accepted labels:

```text
fake
real
misleading
satire
uncertain
```

Feedback flow:

```text
prediction shown in UI
-> user/admin selects human label
-> save to review_labels table
-> later use labels for offline retraining
```

The app does not automatically retrain. It stores reviewed labels for future offline retraining.

## Trending News / GNews Exclusion

GNews/RSS trending news is display-only.

```text
GNews/RSS
-> fetch trending news
-> classify title + description for UI display
-> no SQLite storage
-> no Pinecone storage
```

Only direct user-analyzed content is stored.

GNews is disabled by default. To enable fetching:

```env
ENABLE_GNEWS=true
```

Even when enabled, GNews/RSS items are not stored in SQLite or Pinecone.

## Useful Health Check

Check Pinecone connection and vector count:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:3000/api/vector-status" | ConvertTo-Json -Depth 10
```

Expected working storage response after at least one analysis:

```json
{
  "enabled": true,
  "connected": true,
  "index": "truth-tell-news",
  "namespace": "__default__"
}
```

