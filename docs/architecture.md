flowchart TD
A[Camera / Video] --> B[Video Capture]
B --> C[Preprocessing]
C --> D[AI Inference]
D --> E[Tracking]
E --> F[ROI / Analytics]
F --> G[Event Engine]
G --> H[Snapshot / Evidence]
H --> I[Event Storage]
I --> J[REST API / JSON / MQTT]
J --> K[Monitoring Interface]
B -.-> M[System Monitoring]
D -.-> M
G -.-> M
M --> M1[Camera status]
M --> M2[CPU / GPU / RAM]
M --> M3[Temperature]
M --> M4[FPS / Latency]
M --> M5[Errors / Logs]
