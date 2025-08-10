# Project Structure

```
harassment-detection/
├── core/
│   ├── __init__.py
│   ├── video/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── webcam.py
│   │   ├── rtsp.py
│   │   └── file.py
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── yolo_detector.py
│   │   └── tracker.py
│   │
│   └── action/
│       ├── __init__.py
│       ├── base.py
│       ├── movinet.py
│       ├── custom.py
│       └── registry.py
│
├── models/
│   ├── yolo/
│   ├── movinet/
│   └── custom/
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   ├── buffer.py
│   └── metrics.py
│
├── app/
│   ├── __init__.py
│   ├── pipeline.py
│   └── main.py
│
├── tests/
│   ├── test_video/
│   ├── test_detection/
│   └── test_action/
│
├── config/
│   ├── default.yaml
│   └── custom.yaml
│
├── requirements.txt
├── setup.py
└── README.md
```