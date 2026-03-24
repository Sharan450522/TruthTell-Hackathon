# Model Workflow and Methods

This document captures the current project approach for multimodal fake-news detection, including fine-tuning datasets, model pipelines, and novelty points for presentation.

## Fine-Tuning Datasets

- CLIP/video pipeline dataset: [Kaggle Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge)
- BERT/text pipeline dataset: [Kaggle Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

## Model Components Used

- **Visual-text feature extraction:** `openai/clip-vit-base-patch32`
- **Audio transcription:** Whisper (`base`)
- **OCR:** Tesseract via `pytesseract`
- **Language explanation (optional):** GPT-2
- **Text classifier:** DistilBERT fine-tuned for Fake/True
- **Serving stack:** Flask + Socket.IO + OpenCV + ffmpeg + yt-dlp

## CLIP-Based Video Pipeline (Fine-Tuned Head)

### Core idea

1. Extract sampled video frames.
2. Extract audio and transcribe speech.
3. OCR sampled frames for on-screen text.
4. Encode visual and textual modalities using CLIP.
5. Fuse features (early concatenation).
6. Train a small classifier head (`FusionClassifier`) on fused embeddings.

### Implementation notes

- `metadata.json` labels are normalized to binary (`FAKE -> 1`, `REAL -> 0`).
- Visual embedding is averaged from sampled frames.
- Text embedding is extracted from `transcript + OCR`.
- Training uses `CrossEntropyLoss` and Adam optimizer.
- Saved weights: `fusion_classifier.pth`.

### Important code corrections applied

- `def __init__(...)` (not `_init_`)
- `if __name__ == "__main__":` (not `_name_ == "_main_"`)
- `super().__init__()` in `FusionClassifier`

## BERT/DistilBERT Text Pipeline

### Dataset and preprocessing

- Merge `True.csv` and `Fake.csv`, assign labels (`True=1`, `Fake=0`).
- Shuffle with fixed seed for reproducibility.
- Clean text (lowercasing, whitespace normalization, non-word filtering).
- Stratified train/test split.

### Training setup

- Base model: `distilbert-base-uncased`
- Dynamic padding via `DataCollatorWithPadding`
- Typical config:
  - `num_train_epochs=3`
  - `learning_rate=2e-5`
  - `weight_decay=0.01`
  - early stopping callback
- Metrics: accuracy + classification report

### Important code corrections applied

- `NewsDataset.__init__`, `__len__`, `__getitem__` must use double underscore names.

## End-to-End Functional Workflows

### A) Recorded Video Analysis (URL)

1. Download and standardize video format.
2. Frame extraction (`OpenCV`).
3. Audio extraction (`MoviePy`) and transcription (Whisper/AssemblyAI path).
4. OCR from representative/sampled frames.
5. CLIP visual + text embeddings and early fusion.
6. Classifier + explanation output.

Output: truthiness prediction, confidence, transcript, OCR, and summary details.

### B) Live Stream Analysis (URL)

1. Resolve stream URL.
2. Segment live stream (`ffmpeg`) into fixed windows.
3. Per-segment transcription + OCR + visual sampling.
4. Emit updates through `Socket.IO`.

Output: rolling confidence updates and frame fragments in near real time.

### C) Local Video Upload

1. Save temporary file.
2. Run recorded video pipeline.
3. Return fused analysis summary and clean temporary artifacts.

### D) Local Audio Upload

1. Save temporary audio.
2. Transcribe speech.
3. Run text classifier/ensemble credibility scoring.

Output: transcript + confidence.

### E) Direct Text Analysis

1. Receive raw text.
2. Run text model / ensemble analyzer.

Output: prediction + confidence.

### F) Article URL Analysis

1. Fetch article HTML.
2. Remove noise (`script/style/nav/ads`-like blocks).
3. Extract cleaned textual content.
4. Run text analysis model.

Output: article text summary + confidence output.

## Methods and Algorithms Summary

- Multimodal representation learning using CLIP encoders
- OCR-assisted context augmentation from frames
- Speech-to-text with Whisper/AssemblyAI
- Early feature fusion (concatenation)
- DistilBERT-based text classification for fact-checking pathway
- Ensemble-style confidence aggregation (where configured)
- Real-time event-driven updates using Socket.IO

## Conference Novelty Positioning

1. **Multimodal integration:** unified use of visual, audio, and textual evidence.
2. **Hybrid operation modes:** both batch (recorded/uploaded) and live-stream processing.
3. **OCR + transcription enrichment:** captures hidden cues from captions/on-screen text.
4. **User-facing explainability:** confidence scores, extracted fragments, and generated reasoning.
5. **Production-oriented design:** web interface + backend services for practical deployment.

## Current Project Note

This reflects the latest modified project approach and workflow currently used in the repository.
