import os
import json
import traceback
import cv2
import pytesseract
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd
from PIL import Image

# --------------------------
# Environment & Paths Setup
# --------------------------
TESSERACT_CMD = r"C:\Users\shara\Downloads\fakenews\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

if os.name == "nt":
    ffmpeg_path = r"C:\Users\shara\Downloads\ffmpeg-7.1.1\bin"
    os.environ["PATH"] = ffmpeg_path + ";" + os.environ.get("PATH", "")
    os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.join(ffmpeg_path, "ffmpeg.exe")
else:
    ffmpeg_path = ""

ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg.exe") if os.name == "nt" else "ffmpeg"
os.environ["FFMPEG_BINARY"] = ffmpeg_exe

# Dataset folder configuration
TRAIN_FOLDER = r"C:\Users\shara\Downloads\fakenews\deepfake-detection-challenge\train"

# --------------------------
# Model & Device Initialization
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
whisper_model = whisper.load_model("base").to(device)


# --------------------------
# Utility Functions
# --------------------------
def load_metadata(folder_path):
    """Load metadata.json from the training folder and return a DataFrame.
    Converts string labels 'FAKE'/'REAL' to integers (1 for FAKE, 0 for REAL).
    """
    metadata_path = os.path.join(folder_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {folder_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    data = []
    for video_file, label in metadata.items():
        video_path = os.path.join(folder_path, video_file)

        if isinstance(label, dict):
            label = label.get("label", None)

        if isinstance(label, str):
            label_lower = label.lower()
            if label_lower == "fake":
                label = 1
            elif label_lower == "real":
                label = 0
            else:
                try:
                    label = int(label)
                except Exception as e:
                    print(f"[WARNING] Unable to convert label '{label}' for {video_file}: {e}")
                    continue

        if label not in [0, 1]:
            print(f"[WARNING] Invalid label for {video_file}: {label}")
            continue

        data.append({"file_path": video_path, "label": label})

    return pd.DataFrame(data)


def extract_frames(video_path, frame_interval=30):
    print("[INFO] Extracting frames from video.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Unable to open video file: {video_path}")

    frames = []
    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % frame_interval == 0:
            frames.append(frame)
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    print(f"[INFO] {len(frames)} frames extracted.")
    return frames


def extract_audio(video_path, audio_path="temp_audio.wav"):
    print("[INFO] Extracting audio from video.")
    clip = VideoFileClip(video_path)

    if clip.audio is None:
        raise Exception("No audio stream found in video.")

    clip.audio.write_audiofile(audio_path, logger=None)
    abs_audio_path = os.path.abspath(audio_path)

    if not os.path.exists(abs_audio_path):
        raise FileNotFoundError(f"Audio file not found at {abs_audio_path}")

    return abs_audio_path


def transcribe_audio(audio_path="temp_audio.wav"):
    print("[INFO] Transcribing audio using Whisper.")
    result = whisper_model.transcribe(audio_path)
    return result.get("text", "")


def perform_ocr_on_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(rgb_frame)


def extract_visual_features(frames):
    print("[INFO] Extracting visual features using CLIP.")
    features = []

    for frame in frames:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        features.append(image_features.squeeze().cpu().numpy())

    return features


def extract_textual_features(text):
    print("[INFO] Extracting textual features using CLIP.")
    if not text.strip():
        text = " "

    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)

    return text_features.squeeze().cpu().numpy()


def early_fusion(visual_feat, textual_feat):
    return np.concatenate((visual_feat, textual_feat))


def process_video_for_training(video_path):
    frames = extract_frames(video_path, frame_interval=30)
    if len(frames) == 0:
        raise Exception("No frames extracted from the video.")

    visual_feats = extract_visual_features(frames[:5])
    avg_visual_feat = np.mean(visual_feats, axis=0)

    audio_file = None
    transcript = ""
    try:
        audio_file = extract_audio(video_path)
        transcript = transcribe_audio(audio_file)
    except Exception as e:
        print(f"[WARNING] Audio processing failed for {video_path}: {e}")

    ocr_texts = [perform_ocr_on_frame(frame) for frame in frames[:5]]
    combined_ocr = " ".join(ocr_texts)

    combined_text = (transcript + " " + combined_ocr).strip()
    textual_feat = extract_textual_features(combined_text)

    if audio_file and os.path.exists(audio_file):
        os.remove(audio_file)

    fused_embedding = early_fusion(avg_visual_feat, textual_feat)
    return fused_embedding


# --------------------------
# FusionClassifier Definition
# --------------------------
class FusionClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=2):
        super(FusionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out)


fusion_classifier = FusionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fusion_classifier.parameters(), lr=1e-4)


# --------------------------
# Training Loop
# --------------------------
def train_model(num_epochs=1):
    try:
        df = load_metadata(TRAIN_FOLDER)
    except Exception as e:
        print("Error loading metadata:", e)
        return

    if "file_path" not in df.columns or "label" not in df.columns:
        print("Metadata must contain 'file_path' and 'label' columns.")
        return

    all_preds = []
    all_labels = []
    losses = []

    fusion_classifier.train()

    for epoch in range(num_epochs):
        epoch_losses = []
        print(f"\n[INFO] Starting Epoch {epoch + 1}/{num_epochs}")

        for _, row in df.iterrows():
            video_path = row["file_path"]
            label_val = row["label"]

            try:
                label = int(label_val)
            except Exception as conv_err:
                print(f"[WARNING] Skipping {video_path} due to label conversion error: {conv_err}")
                continue

            if not os.path.exists(video_path):
                print(f"[WARNING] File not found: {video_path}")
                continue

            try:
                fused_embedding = process_video_for_training(video_path)
            except Exception as e:
                print(f"[WARNING] Skipping {video_path} due to error: {e}")
                continue

            fused_tensor = torch.tensor(fused_embedding, dtype=torch.float32).unsqueeze(0).to(device)
            target = torch.tensor([label], dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = fusion_classifier(fused_tensor)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pred = outputs.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(label)
            print(f"Processed {video_path} | Loss: {loss.item():.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        losses.append(avg_loss)
        print(f"[INFO] Epoch {epoch + 1} average loss: {avg_loss:.4f}")

    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        if len(all_labels) == 0:
            raise ValueError("No valid samples processed; metrics unavailable.")

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    except Exception as metric_e:
        print("Error calculating metrics:", metric_e)
        acc = prec = rec = f1 = None

    torch.save(fusion_classifier.state_dict(), "fusion_classifier.pth")
    print("[INFO] Training completed and model saved to fusion_classifier.pth")
    print("Metrics:")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
    print(f"Losses per epoch: {losses}")


if __name__ == "__main__":
    try:
        num_epochs = 5
        train_model(num_epochs=num_epochs)
    except Exception:
        print("Training failed with error:")
        traceback.print_exc()
