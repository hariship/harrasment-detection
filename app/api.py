"""
FastAPI ASGI application for Harassment Detection System
Provides REST API and WebSocket endpoints for real-time video processing
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import cv2
import numpy as np
import asyncio
import json
import base64
from typing import Optional, Dict, Any
import io
from PIL import Image
import yaml

from app.pipeline import HarassmentDetectionPipeline
from app.simple_api import router as stats_router, update_stats


# Global pipeline instance
pipeline: Optional[HarassmentDetectionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup pipeline on startup/shutdown"""
    global pipeline
    
    # Load config
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    print("Initializing harassment detection pipeline...")
    pipeline = HarassmentDetectionPipeline(config)
    if not pipeline.initialize():
        print("Warning: Pipeline initialization failed")
    
    yield
    
    # Cleanup
    if pipeline:
        pipeline.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Harassment Detection API",
    description="Real-time harassment detection system for surveillance cameras",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include stats router
app.include_router(stats_router)


@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "name": "Harassment Detection System",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "POST /detect/image": "Detect harassment in uploaded image",
            "POST /detect/base64": "Detect harassment in base64 image",
            "WS /ws": "WebSocket for real-time video streaming",
            "GET /stream": "Video stream endpoint",
            "GET /demo": "Demo web interface"
        }
    }


@app.get("/health")
async def health_check():
    """Check if system is operational"""
    global pipeline
    return {
        "status": "healthy" if pipeline else "unhealthy",
        "pipeline_initialized": pipeline is not None,
        "components": {
            "video_source": pipeline.video_source is not None if pipeline else False,
            "person_detector": pipeline.person_detector is not None if pipeline else False,
            "action_detector": pipeline.action_detector is not None if pipeline else False
        }
    }


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect harassment in uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
    
    Returns:
        Detection results with bounding boxes and actions
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    # Process frame
    results = pipeline.process_frame(frame)
    
    return {
        "success": True,
        "detections": results['persons'],
        "harassment_detected": results['harassment_detected'],
        "alerts": results['alerts']
    }


@app.post("/detect/base64")
async def detect_base64(data: Dict[str, str]):
    """
    Detect harassment in base64 encoded image
    
    Args:
        data: JSON with 'image' field containing base64 string
    
    Returns:
        Detection results
    """
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process frame
        results = pipeline.process_frame(frame)
        
        return {
            "success": True,
            "detections": results['persons'],
            "harassment_detected": results['harassment_detected'],
            "alerts": results['alerts']
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming
    Client sends video frames, server responds with detection results
    """
    global pipeline
    await websocket.accept()
    
    if not pipeline:
        await websocket.send_json({"error": "Pipeline not initialized"})
        await websocket.close()
        return
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Decode base64 frame
            image_data = base64.b64decode(frame_data['frame'])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Process frame
                results = pipeline.process_frame(frame)
                
                # Send results back
                await websocket.send_json({
                    "detections": results['persons'],
                    "harassment_detected": results['harassment_detected'],
                    "alerts": results['alerts'],
                    "fps": results['fps']
                })
            else:
                await websocket.send_json({"error": "Invalid frame"})
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


async def generate_frames():
    """Generate video frames for streaming"""
    global pipeline
    if not pipeline or not pipeline.video_source:
        return
    
    while True:
        ret, frame = pipeline.video_source.read_frame()
        if not ret:
            break
        
        # Process frame
        results = pipeline.process_frame(frame)
        
        # Update global stats
        update_stats(results)
        
        # Visualize results
        display_frame = pipeline.visualize_results(frame, results)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        await asyncio.sleep(0.1)  # ~10 FPS for better performance


@app.get("/stream")
async def video_stream():
    """
    Video streaming endpoint
    Returns MJPEG stream for browser viewing
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/demo", response_class=HTMLResponse)
async def demo_interface():
    """Simple web interface for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gesture-Based Threat Detection System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .container {
                display: flex;
                gap: 20px;
                margin-top: 20px;
            }
            .video-container {
                flex: 1;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            #videoFeed {
                width: 100%;
                border-radius: 4px;
            }
            .info-container {
                width: 300px;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .status {
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 10px;
            }
            .status.alert {
                background-color: #ffebee;
                color: #c62828;
                border: 1px solid #ef5350;
            }
            .status.safe {
                background-color: #e8f5e9;
                color: #2e7d32;
                border: 1px solid #66bb6a;
            }
            .detection-list {
                margin-top: 20px;
            }
            .detection-item {
                padding: 8px;
                background: #f9f9f9;
                margin-bottom: 8px;
                border-radius: 4px;
                border-left: 3px solid #2196f3;
            }
            .controls {
                margin-top: 20px;
                text-align: center;
            }
            button {
                padding: 10px 20px;
                margin: 5px;
                border: none;
                border-radius: 4px;
                background-color: #2196f3;
                color: white;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #1976d2;
            }
            button:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
        </style>
    </head>
    <body>
        <h1>👁️ Gesture-Based Threat Detection</h1>
        
        <div class="container">
            <div class="video-container">
                <h2>Live Feed</h2>
                <img id="videoFeed" src="/stream" alt="Video Stream">
                <div class="controls">
                    <button onclick="location.reload()">Refresh Stream</button>
                    <button onclick="takeSnapshot()">Take Snapshot</button>
                </div>
            </div>
            
            <div class="info-container">
                <h2>Detection Status</h2>
                <div id="status" class="status safe">
                    👀 Monitoring for Threatening Gestures
                </div>
                
                <div class="detection-list">
                    <h3>Gesture Analysis</h3>
                    <div id="detections">
                        <div class="detection-item">Monitoring...</div>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <h3>Statistics</h3>
                    <div id="stats">
                        <p>FPS: <span id="fps">--</span></p>
                        <p>Persons Detected: <span id="personCount">0</span></p>
                        <p>Gesture Alerts: <span id="alertCount">0</span></p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // WebSocket connection for real-time updates
            let ws = null;
            let alertCount = 0;
            
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateStatus(data);
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
                
                ws.onclose = function() {
                    setTimeout(connectWebSocket, 5000); // Reconnect after 5 seconds
                };
            }
            
            function updateStatus(data) {
                // Update status
                const statusEl = document.getElementById('status');
                if (data.harassment_detected) {
                    statusEl.className = 'status alert';
                    statusEl.textContent = '⚠️ HARASSMENT DETECTED';
                    alertCount++;
                } else {
                    statusEl.className = 'status safe';
                    statusEl.textContent = '✓ System Active - No Threats Detected';
                }
                
                // Update statistics
                document.getElementById('fps').textContent = data.fps ? data.fps.toFixed(1) : '--';
                document.getElementById('personCount').textContent = data.detections ? data.detections.length : 0;
                document.getElementById('alertCount').textContent = alertCount;
                
                // Update detections
                if (data.detections && data.detections.length > 0) {
                    const detectionsHtml = data.detections.map(d => 
                        `<div class="detection-item">
                            Person ${d.id || 'Unknown'}: ${d.actions.join(', ') || 'Normal'}
                        </div>`
                    ).join('');
                    document.getElementById('detections').innerHTML = detectionsHtml;
                }
            }
            
            function takeSnapshot() {
                const img = document.getElementById('videoFeed');
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                
                // Download snapshot
                const link = document.createElement('a');
                link.download = 'snapshot_' + new Date().getTime() + '.png';
                link.href = canvas.toDataURL();
                link.click();
            }
            
            // Connect WebSocket on load
            connectWebSocket();
            
            // Also poll stats API as backup
            setInterval(updateStatsFromAPI, 2000); // Update every 2 seconds
            
            async function updateStatsFromAPI() {
                try {
                    const response = await fetch('/stats');
                    const data = await response.json();
                    
                    // Update stats display
                    document.getElementById('personCount').textContent = data.persons_detected || 0;
                    document.getElementById('fps').textContent = data.fps ? data.fps.toFixed(1) : '--';
                    document.getElementById('alertCount').textContent = data.alert_count || 0;
                    
                    // Update status with gesture focus
                    const statusEl = document.getElementById('status');
                    if (data.harassment_detected) {
                        statusEl.className = 'status alert';
                        statusEl.textContent = '🚨 THREATENING GESTURE DETECTED';
                    } else {
                        statusEl.className = 'status safe';
                        statusEl.textContent = '👀 Monitoring for Threatening Gestures';
                    }
                    
                    // Update detections
                    if (data.last_detections && data.last_detections.length > 0) {
                        const detectionsHtml = data.last_detections.map(d => 
                            `<div class="detection-item">
                                Person ${d.id || 'Unknown'}: ${d.actions ? d.actions.join(', ') : 'Normal'}
                            </div>`
                        ).join('');
                        document.getElementById('detections').innerHTML = detectionsHtml;
                    } else if (data.persons_detected > 0) {
                        document.getElementById('detections').innerHTML = 
                            `<div class="detection-item">${data.persons_detected} person(s) detected</div>`;
                    } else {
                        document.getElementById('detections').innerHTML = 
                            '<div class="detection-item">Monitoring...</div>';
                    }
                } catch (error) {
                    console.error('Stats update error:', error);
                }
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)