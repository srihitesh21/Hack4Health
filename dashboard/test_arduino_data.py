#!/usr/bin/env python3
"""
Test script to simulate Arduino data for testing the dashboard
Run this script to send simulated sensor data to the dashboard
"""

import serial
import time
import random
import json

def simulate_arduino_data():
    """Simulate Arduino sending sensor data"""
    
    # Try to find available serial ports
    import glob
    ports = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/ttyACM*') + glob.glob('COM*')
    
    if not ports:
        print("No serial ports found. Creating a virtual port for testing...")
        print("Note: You'll need to manually connect to Arduino in the dashboard.")
        return
    
    # Use the first available port
    port = ports[0]
    print(f"Using port: {port}")
    
    try:
        # Open serial connection
        ser = serial.Serial(port, 9600, timeout=1)
        print(f"Connected to {port}")
        print("Sending simulated sensor data...")
        print("Press Ctrl+C to stop")
        
        # Simulate sensor data
        base_temp = 25.0
        base_humidity = 60.0
        base_light = 500
        base_pressure = 1013.25
        
        while True:
            # Generate realistic sensor variations
            temperature = base_temp + random.uniform(-2, 2)
            humidity = base_humidity + random.uniform(-5, 5)
            humidity = max(0, min(100, humidity))  # Keep within 0-100%
            light = base_light + random.uniform(-50, 50)
            light = max(0, min(1023, light))  # Keep within 0-1023
            pressure = base_pressure + random.uniform(-5, 5)
            
            # Send data in CSV format (recommended)
            data_line = f"{temperature:.1f},{humidity:.1f},{light:.0f},{pressure:.2f}\n"
            ser.write(data_line.encode())
            
            print(f"Sent: {data_line.strip()}")
            time.sleep(1)  # Send data every second
            
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        print("Make sure Arduino is connected and not being used by another application")
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        if 'ser' in locals():
            ser.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Arduino Data Simulator")
    print("This script simulates Arduino sensor data for testing the dashboard")
    print("Make sure the dashboard is running first (python arduino_dashboard.py)")
    print()
    
    simulate_arduino_data() 