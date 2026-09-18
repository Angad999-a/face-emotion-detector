# Design Diagrams

These render natively on GitHub. Paste the same Mermaid blocks into your project report (Word/PDF), or export as PNG (see note at bottom) for the report if your report tool doesn't support Mermaid.

## 1. System Architecture Diagram

```mermaid
flowchart LR
    A[Input Source<br/>Webcam / Video / Images] --> B[Video Stream Handler]
    B --> C[Face Detector Module<br/>OpenCV DNN]
    C --> D[Emotion Classifier Module<br/>Pretrained CNN - FER2013]
    D --> E[Analytics Engine<br/>Aggregation]
    E --> F[CSV Log]
    E --> G[Summary Chart]
    E --> H[Console Report]
    C --> I[Annotated Frame Output]
    D --> I
    B --> J[Logger]
    C --> J
    D --> J
```

## 2. Process / Workflow Diagram

```mermaid
flowchart TD
    Start([Start]) --> Init[Parse CLI args & load config]
    Init --> Open[Open input source]
    Open --> Loop{Frames remaining?}
    Loop -- Yes --> Read[Read next frame]
    Read --> Detect[Detect faces in frame]
    Detect --> HasFace{Face found?}
    HasFace -- No --> Log1[Log 'no face' event]
    Log1 --> Loop
    HasFace -- Yes --> Crop[Crop each face region]
    Crop --> Classify[Classify emotion per face]
    Classify --> Annotate[Draw box + label on frame]
    Annotate --> Record[Append record to CSV log]
    Record --> Loop
    Loop -- No --> Aggregate[Aggregate session results]
    Aggregate --> Chart[Generate summary chart]
    Chart --> Report[Print console summary]
    Report --> End([End])
```

## 3. Use Case Diagram

```mermaid
flowchart LR
    User((User))
    UC1([Run live webcam detection])
    UC2([Analyze video file])
    UC3([Analyze image folder])
    UC4([View session report])
    UC5([Configure thresholds])

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
```

## 4. Class / Component Diagram

```mermaid
classDiagram
    class CLIController {
        +parse_args()
        +run()
    }
    class VideoStream {
        +source: str
        +open()
        +read_frame()
        +release()
    }
    class FaceDetector {
        +model
        +confidence_threshold: float
        +detect(frame) List~BoundingBox~
    }
    class EmotionClassifier {
        +model
        +confidence_threshold: float
        +classify(face_crop) EmotionResult
    }
    class AnalyticsEngine {
        +records: List~Record~
        +add_record(record)
        +generate_chart()
        +generate_summary()
    }
    class LoggerConfig {
        +setup_logger()
    }

    CLIController --> VideoStream
    CLIController --> FaceDetector
    CLIController --> EmotionClassifier
    CLIController --> AnalyticsEngine
    CLIController --> LoggerConfig
    FaceDetector --> AnalyticsEngine : detections
    EmotionClassifier --> AnalyticsEngine : classifications
```

## 5. Sequence Diagram (single frame processing)

```mermaid
sequenceDiagram
    participant U as User
    participant M as main.py
    participant VS as VideoStream
    participant FD as FaceDetector
    participant EC as EmotionClassifier
    participant AE as AnalyticsEngine

    U->>M: run --mode video --input file.mp4
    M->>VS: open(source)
    loop for each frame
        M->>VS: read_frame()
        VS-->>M: frame
        M->>FD: detect(frame)
        FD-->>M: [bounding_boxes]
        loop for each face
            M->>EC: classify(face_crop)
            EC-->>M: emotion, confidence
            M->>AE: add_record(...)
        end
        M->>M: draw annotations, save frame
    end
    M->>AE: generate_chart()
    M->>AE: generate_summary()
    AE-->>U: summary report + chart + CSV
```

---
### Note on exporting to PNG for the PDF report
If your report tool doesn't render Mermaid, paste any block above into the free [Mermaid Live Editor](https://mermaid.live), export as PNG/SVG, and embed the image in your report instead.
