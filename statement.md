# Problem Statement

## Problem
Understanding human emotional response at scale — in classrooms, retail environments, user-testing sessions, or public safety monitoring — currently relies on manual observation, which is slow, subjective, and doesn't scale beyond a handful of people. There is a need for an automated, lightweight tool that can detect faces in a video/image stream and classify the emotional expression of each face in real time, without requiring specialized hardware or custom model training.

## Scope
This project delivers a command-line computer vision system that:
- Accepts input from a live webcam, a pre-recorded video file, or a folder of static images
- Detects all faces present in each frame using a pretrained face-detection model
- Classifies each detected face into one of seven standard emotion categories using a pretrained CNN (trained on FER2013)
- Logs every detection with timestamp, bounding box, and predicted emotion to a CSV file
- Produces an end-of-session summary (emotion distribution chart + console report)

**Out of scope:** training a custom emotion model from scratch, multi-camera synchronization, real-time face re-identification/tracking across long video sessions, and cloud deployment. These are noted as future enhancements.

## Target Users
- Students/researchers studying human-computer interaction or affective computing
- Educators wanting a lightweight tool to gauge classroom engagement in recorded sessions
- UX researchers reviewing recorded usability-testing sessions for emotional response
- Anyone wanting a hands-on, runnable example of a pretrained-model-based CV pipeline

## High-Level Features
1. **Face Detection** — locate all faces in a frame with bounding boxes (pretrained OpenCV DNN detector)
2. **Emotion Classification** — classify each face into angry / disgust / fear / happy / sad / surprise / neutral (pretrained FER2013-based CNN)
3. **Analytics & Reporting** — aggregate results across a session into a CSV log, summary chart, and console report

## Success Criteria
- Correctly detects faces in well-lit, front-facing sample images/video with reasonable accuracy
- Correctly classifies emotion for detected faces with accuracy comparable to the pretrained model's published benchmarks (~65-70% on FER2013 for models like this)
- Runs end-to-end from the command line on all three input modes without crashing on edge cases (no face present, corrupted frame, empty folder)
- Produces a usable CSV log and summary chart every run
