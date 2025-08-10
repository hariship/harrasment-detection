"""
Simple API endpoint to get current detection stats
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import asyncio
import json

# Global stats storage
current_stats = {
    "persons_detected": 0,
    "harassment_detected": False,
    "fps": 0.0,
    "last_detections": [],
    "alert_count": 0
}

def update_stats(results: Dict[str, Any]):
    """Update global stats with latest detection results"""
    global current_stats
    
    current_stats["persons_detected"] = len(results.get('persons', []))
    current_stats["harassment_detected"] = results.get('harassment_detected', False)
    current_stats["fps"] = results.get('fps', 0.0)
    current_stats["last_detections"] = results.get('persons', [])
    
    if results.get('harassment_detected'):
        current_stats["alert_count"] += 1

def get_current_stats() -> Dict[str, Any]:
    """Get current detection statistics"""
    return current_stats.copy()

# API Router
router = APIRouter()

@router.get("/stats")
async def get_stats():
    """Get real-time detection statistics"""
    return get_current_stats()

@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check with current stats"""
    stats = get_current_stats()
    return {
        "status": "healthy",
        "current_detections": stats["persons_detected"],
        "harassment_active": stats["harassment_detected"],
        "fps": stats["fps"],
        "total_alerts": stats["alert_count"]
    }