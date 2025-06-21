#!/bin/bash

# Arduino Dashboard Startup Script

echo "🚀 Starting Arduino Dashboard..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check if requirements are installed
if ! python3 -c "import flask, flask_socketio, serial" &> /dev/null; then
    echo "📦 Installing Python dependencies..."
    pip3 install -r requirements.txt
    echo ""
fi

# Start the dashboard
echo "🌐 Starting web server..."
echo "📱 Open your browser and go to: http://localhost:5000"
echo "🔌 Connect your Arduino and click 'Connect Arduino' in the dashboard"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 arduino_dashboard.py 