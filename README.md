# IIS_Project_2025fall

Emotion-Adaptive TRPG Dungeon Master with real-time facial expression perception and story-grounded narration.

## Overview
- Perception: Webcam → DINOv3 features → SVM classifier (excited / nervous / confused).
- EmotionMeter: Aggregates emotion over time and outputs low/medium/high levels.
- Narrative: Story beats + retrieval from a PDF script; emotion-driven controls for density/risk/pacing.
- HUD: Live camera view, radar meter, and strategy hints.

## Requirements
- Python 3.9+
- OpenCV, NumPy, scikit-learn, joblib, transformers, torch
- MediaPipe (for face detection in `dataset/mydata/process_mydata_videos.py`)
- Furhat Realtime API (for `main.py`)
- Gemini API key for LLM (`GEMINI_API_KEY`)

## Project Structure (key files)
- `main.py`: Runs the full DM loop with Furhat + emotion-driven narration.
- `perception.py`: Emotion perception pipeline (DINOv3 + SVM + confidence gate).
- `emotion_state.py`: EmotionMeter (decay, levels, deep-help mode).
- `narrative.py`: Gemini DM prompting with emotion controls.
- `story_manager.py`: Story beat controller + retrieval from PDF.
- `visualization.py`: HUD.
- `train_classifier.py`: Train 3-class model on personal data.
- `test_emotion_realtime.py`: Realtime emotion test with webcam.
- `dataset/mydata/process_mydata_videos.py`: Extract face frames from labeled videos.

## Setup
1) Install dependencies (example):
```bash
pip install opencv-python numpy scikit-learn joblib torch transformers mediapipe
```
2) Set Gemini API key (if using LLM):
```bash
set GEMINI_API_KEY=your_key_here
```

## Data Preparation (personal videos)
Place labeled videos in:
```
dataset/mydata/video/
```
Filename should include the label, e.g. `excited_01.mp4`, `nervous2.mov`, `confused_take3.avi`.

Then run:
```bash
python .\dataset\mydata\process_mydata_videos.py
```
Output images go to:
```
dataset/mydata/excited
dataset/mydata/nervous
dataset/mydata/confused
```

## Train Model (personal data only)
```bash
python .\train_classifier.py
```
This outputs:
```
dinov3_svm_3class.joblib
```

## Realtime Emotion Test
```bash
python .\test_emotion_realtime.py
```
Press `q` to exit.

## Run the Full DM System
```bash
python .\main.py
```
Notes:
- Requires Furhat Realtime API.
- Emotion capture occurs while the DM speaks.
- The model file must exist at `dinov3_svm_3class.joblib`.

## Notes
- If no face is detected, the frame is skipped.
- Confidence gate marks results as `uncertain` when the top emotion is weak or too close to the runner-up.
