#!/bin/bash

# Simple curl tests for the Harassment Detection API
# Run this to quickly test your running server

BASE_URL="http://localhost:8000"

echo "🧪 Simple API Tests with curl"
echo "================================"
echo

# Test 1: Health check
echo "1️⃣ Testing health endpoint..."
curl -s "$BASE_URL/health" | python3 -m json.tool
echo
echo

# Test 2: Root endpoint
echo "2️⃣ Testing root endpoint..."
curl -s "$BASE_URL/" | python3 -m json.tool
echo
echo

# Test 3: Check if stream is accessible
echo "3️⃣ Testing stream endpoint (first few bytes)..."
timeout 2 curl -s "$BASE_URL/stream" | head -c 200
echo
echo "... (stream continues)"
echo
echo

# Test 4: Check demo page
echo "4️⃣ Testing demo page (HTML length)..."
DEMO_LENGTH=$(curl -s "$BASE_URL/demo" | wc -c)
echo "Demo page HTML length: $DEMO_LENGTH characters"
if [ $DEMO_LENGTH -gt 1000 ]; then
    echo "✅ Demo page looks good!"
else
    echo "❌ Demo page might have issues"
fi
echo

echo "🌐 Browser URLs to test:"
echo "   Demo Interface: $BASE_URL/demo"
echo "   API Docs:       $BASE_URL/docs"
echo "   Video Stream:   $BASE_URL/stream"
echo
echo "💡 For image upload tests, run: python test_api.py"