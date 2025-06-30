#!/bin/bash

# Health Monitoring Dashboard Startup Script
# Unified Dashboard with Health Monitoring and Heat Stroke Assessment

echo "=========================================="
echo "Health Monitoring Dashboard"
echo "Unified Interface with Heat Stroke Assessment"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.7+"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create data directory
mkdir -p data

echo ""
echo "=========================================="
echo "Starting Health Monitoring Dashboard..."
echo "=========================================="
echo ""
echo "Dashboard Features:"
echo "✓ Real-time Health Monitoring"
echo "✓ Heart Rate & Temperature Charts"
echo "✓ Activity & Humidity Tracking"
echo "✓ Heat Stroke Risk Assessment"
echo "✓ User Profile Management"
echo "✓ Assessment History"
echo "✓ Emergency Alerts"
echo ""
echo "Access the dashboard at: http://localhost:5000"
echo "Health check endpoint: http://localhost:5000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the dashboard
python3 arduino_dashboard.py 