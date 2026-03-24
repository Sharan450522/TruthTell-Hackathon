import os
import cv2
import pytesseract
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
import traceback
import base64
import platform

# --- FFmpeg and Tesseract Setup ---
if platform.system() == "Windows":
    ffmpeg_path = r"C:\Program Files\ffmpeg\bin"
    os.environ["PATH"] = ffmpeg_path + ";" + os.environ.get("PATH", "")
    os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.join(ffmpeg_path, "ffmpeg.exe")
    ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg.exe")
else:
    ffmpeg_exe = "ffmpeg"

os.environ["FFMPEG_BINARY"] = ffmpeg_exe
if platform.system() == "Windows" and not os.path.exists(ffmpeg_exe):
    raise FileNotFoundError(f"ffmpeg not found at {ffmpeg_exe}. Please verify the installation path.")

# Set Tesseract path (adjust as needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Model and Device Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
whisper_model = whisper.load_model("base").to(device)
if gpt2_model.config.pad_token_id is None:
    gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id


class VideoProcessor:
    """
    Processes video files for fake news detection.
    Extracts frames, audio transcription, OCR, and visual/textual features.
    Generates a GPT-2 inference and returns a result summary including thumbnails.
    Optionally, it computes a confidence score via an analyze_text function.
    """

    def __init__(self, cache_manager=None, analyze_text_func=None):
        self.cache = cache_manager
        self.frame_interval = 30
        self.analyze_text = analyze_text_func

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        success, frame = cap.read()
        while success:
            if frame_count % self.frame_interval == 0:
                frames.append(frame)
            success, frame = cap.read()
            frame_count += 1
        cap.release()
        return frames

    def extract_audio(self, video_path, audio_path="temp_audio.wav"):
        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, logger=None)
            abs_audio_path = os.path.abspath(audio_path)
            if not os.path.exists(abs_audio_path):
                raise FileNotFoundError(f"Audio file not found at {abs_audio_path}")
            return abs_audio_path
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {e}")

    def transcribe_audio(self, audio_path):
        try:
            result = whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            raise RuntimeError(f"Audio transcription failed: {e}")

    def perform_ocr(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_string(rgb_frame)

    def extract_visual_features(self, frames):
        features = []
        for frame in frames:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
            features.append(image_features.squeeze().cpu().numpy())
        return features

    def extract_textual_features(self, text):
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
        return text_features.squeeze().cpu().numpy()

    def early_fusion(self, visual_feat, textual_feat):
        return np.concatenate((visual_feat, textual_feat))

    def generate_inference(self, augmented_context, max_length=150):
        prompt = (
            "Based on the following context, determine if the video news is fake and explain why.\n"
            f"Context: {augmented_context}\n"
            "Answer (include a verdict and explanation):"
        )
        inputs = gpt2_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = gpt2_model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            temperature=0.7,
        )
        generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def process_video(self, video_path):
        """
        Process a video by extracting one representative frame, transcript, OCR,
        fused features, and a GPT-2 inference.
        """
        try:
            cache_key = f"video_{os.path.basename(video_path)}"
            if self.cache:
                self.cache.set("video", cache_key, video_path, ttl=600)

            frames = self.extract_frames(video_path)
            if not frames:
                raise ValueError("No frames extracted from video.")
            num_frames = len(frames)
            rep_frame = frames[0]

            audio_file = self.extract_audio(video_path)
            transcript = self.transcribe_audio(audio_file)
            if os.path.exists(audio_file):
                os.remove(audio_file)

            ocr_text = self.perform_ocr(rep_frame)
            visual_features = self.extract_visual_features([rep_frame])
            if not visual_features:
                raise ValueError("No visual features extracted.")
            avg_visual_feature = visual_features[0]

            combined_text = transcript + " " + ocr_text
            textual_feature = self.extract_textual_features(combined_text)
            fused_embedding = self.early_fusion(avg_visual_feature, textual_feature)

            augmented_context = (
                f"Fused feature summary (mean value: {np.mean(fused_embedding):.4f}); "
                f"Transcript (first 100 chars): {transcript[:100]}...; OCR (first 100 chars): {ocr_text[:100]}..."
            )
            llm_output = self.generate_inference(augmented_context)

            ret, buffer = cv2.imencode(".jpg", rep_frame)
            if not ret:
                raise ValueError("Failed to encode representative frame.")
            rep_frame_base64 = base64.b64encode(buffer).decode("utf-8")

            visual_mean = np.mean(avg_visual_feature)
            result_summary = {
                "num_frames_extracted": num_frames,
                "transcript": transcript,
                "ocr_text": ocr_text,
                "avg_visual_feature": avg_visual_feature.tolist(),
                "visual_feature_mean": float(round(visual_mean, 4)),
                "textual_feature": textual_feature.tolist(),
                "fused_embedding": fused_embedding.tolist(),
                "augmented_context": augmented_context,
                "llm_inference": llm_output,
                "representative_frame": rep_frame_base64,
            }

            if self.cache:
                self.cache.delete("video", cache_key)

            return result_summary
        except Exception as e:
            traceback.print_exc()
            raise e

    def process_recorded_video(self, video_path):
        """
        Process recorded video with frame thumbnails, transcript, OCR, fused features,
        GPT-2 inference, and optional text-confidence analysis.
        """
        try:
            frames = self.extract_frames(video_path)
            if not frames:
                raise ValueError("No frames extracted from video.")
            num_frames = len(frames)

            audio_file = self.extract_audio(video_path)
            transcript = self.transcribe_audio(audio_file)
            if os.path.exists(audio_file):
                os.remove(audio_file)

            ocr_texts = []
            visual_features_list = []
            fragments = []
            for frame in frames:
                thumb = cv2.resize(frame, (160, 90))
                ret, buffer = cv2.imencode(".jpg", thumb)
                if ret:
                    frag_base64 = base64.b64encode(buffer).decode("utf-8")
                    fragments.append(frag_base64)

                ocr_text = self.perform_ocr(frame)
                ocr_texts.append(ocr_text)
                features = self.extract_visual_features([frame])
                if features:
                    visual_features_list.append(features[0])

            combined_ocr_text = " ".join(ocr_texts)
            avg_visual_feature = np.mean(np.array(visual_features_list), axis=0)
            textual_feature = self.extract_textual_features(combined_ocr_text)
            fused_embedding = self.early_fusion(avg_visual_feature, textual_feature)

            augmented_context = (
                f"Fused feature summary (mean value: {np.mean(fused_embedding):.4f}); "
                f"Transcript (first 100 chars): {transcript[:100]}...; "
                f"OCR (first 100 chars): {combined_ocr_text[:100]}..."
            )
            llm_output = self.generate_inference(augmented_context)

            rep_frame = frames[0]
            ret, buffer = cv2.imencode(".jpg", rep_frame)
            if not ret:
                raise ValueError("Failed to encode representative frame.")
            rep_frame_base64 = base64.b64encode(buffer).decode("utf-8")

            visual_mean = np.mean(avg_visual_feature)

            if self.analyze_text is not None:
                confidence_analysis = self.analyze_text(transcript)
                confidence = confidence_analysis.get("confidence", 0)
            else:
                confidence = 0
                confidence_analysis = {}

            video_summary = {
                "num_frames_extracted": num_frames,
                "ocr_text": combined_ocr_text,
                "avg_visual_feature": avg_visual_feature.tolist(),
                "visual_feature_mean": float(round(visual_mean, 4)),
                "textual_feature": textual_feature.tolist(),
                "fused_embedding": fused_embedding.tolist(),
                "augmented_context": augmented_context,
                "llm_inference": llm_output,
                "representative_frame": rep_frame_base64,
                "video_fragments": fragments,
                "confidence": confidence,
                "analysis": confidence_analysis,
            }

            return {"transcript": transcript, "video_summary": video_summary}
        except Exception as e:
            traceback.print_exc()
            raise e
