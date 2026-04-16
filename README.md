# Truth Tell Fake News

Fake-news analysis app for article URLs, YouTube recorded videos, audio files, video files, and live streams.

## Setup

Open PowerShell and run:

```powershell
cd D:\truth-tell-fake-news-backup\truth-tell-fake-news-backup

py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists, you can skip:

```powershell
py -3 -m venv .venv
```

## Run The App

```powershell
cd D:\truth-tell-fake-news-backup\truth-tell-fake-news-backup
.\.venv\Scripts\python.exe app.py
```

Then open:

```text
http://localhost:3000
```

or:

```text
http://127.0.0.1:3000
```

## Pinecone Integrated Embedding Setup

Use Pinecone integrated embedding so you do not need OpenAI billing.

In Pinecone, create the index with:

```text
Model: llama-text-embed-v2
Modality: Text
Vector type: Dense
Dimension: 1024
Metric: cosine
Field map: text
```

Recommended index name:

```text
truth-tell-news
```

Recommended namespace:

```text
__default__
```

## Required `.env`

Create or update `.env`:

```env
PORT=3000
SECRET_KEY=change_this_to_any_random_secret

MODEL_VERSION=ensemble_v1
TRUTH_TELL_DB_PATH=data/truth_tell.db

ASSEMBLYAI_API_KEY=PASTE_YOUR_ASSEMBLYAI_API_KEY_HERE

PINECONE_API_KEY=PASTE_YOUR_PINECONE_API_KEY_HERE
PINECONE_INDEX_NAME=truth-tell-news
PINECONE_NAMESPACE=__default__
PINECONE_EMBEDDING_PROVIDER=pinecone
PINECONE_TEXT_FIELD=text
PINECONE_UPSERT_BATCH_SIZE=32

OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=

ENABLE_GNEWS=false
GNEWS_API_KEY=
RAPIDAPI_KEY=

YT_DLP_COOKIES_FROM_BROWSER=chrome
YT_DLP_USE_BROWSER_COOKIES_FOR_LIVE=false
YT_DLP_COOKIES_FILE=
YT_DLP_EXTRACTOR_ARGS=
YT_DLP_LIVE_AUDIO_FORMAT=ba/bestaudio/best
YT_DLP_LIVE_VIDEO_FORMAT=bv*[height<=480]/best[height<=480]/best
YT_DLP_JS_RUNTIME=

HF_TOKEN=
```

## Test Text Analysis And Pinecone Storage

Keep the Flask app running in one terminal.

Open another PowerShell terminal and run:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:3000/api/analyze-text" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"The government announced a new education policy today."}' |
ConvertTo-Json -Depth 10
```

You want to see:

```json
"vector_storage": {
  "embedding_provider": "pinecone",
  "enabled": true,
  "index": "truth-tell-news",
  "namespace": "__default__",
  "pinecone_text_field": "text",
  "upserted": 1
}
```

This means:

```text
text -> Pinecone integrated embedding -> vector stored
```

No OpenAI API key is needed.

## Check Vector Count

Run:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:3000/api/vector-status" | ConvertTo-Json -Depth 10
```

After at least one successful analysis, you should see:

```json
"total_vector_count": 1
```

or:

```json
"namespaces": {
  "__default__": {
    "vector_count": 1
  }
}
```

Pinecone dashboard can take a few seconds to refresh.

## Important Notes

The classifier can return `fake`, `real`, or `uncertain` even without Pinecone because it uses local model files:

```text
saved_models/pipeline.pkl
saved_models/ensemble_model.pkl
```

Pinecone storage is separate from classification.

With the current setup, Pinecone storage uses this chain:

```text
PINECONE_API_KEY valid
-> Pinecone receives text through upsert_records
-> Pinecone creates embedding with llama-text-embed-v2
-> Pinecone stores vector
```

OpenAI embedding is disabled in this setup.

## GNews / Trending News

GNews is disabled by default:

```env
ENABLE_GNEWS=false
```

Even if enabled, GNews/RSS trending news is display-only:

```text
GNews/RSS
-> classify title + description for UI display
-> no SQLite storage
-> no Pinecone storage
```

Only direct user-analyzed content is stored.

