import serial
import csv
from datetime import datetime

# === Configuration ===
SERIAL_PORT = '/dev/cu.usbmodem11301'  # Update this based on your system
BAUD_RATE = 9600

# === Generate filename with timestamp ===
filename = f"arduino_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# === Open serial connection ===
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except serial.SerialException as e:
    print(f"❌ Could not open serial port {SERIAL_PORT}: {e}")
    exit(1)

# === Open CSV and start logging ===
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp(ms)", "SkinTemp(C)", "IR_Value"])  # CSV header

    print(f"📡 Logging to {filename}... Press Ctrl+C to stop.")
    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line.count(',') == 2:
                parts = line.split(',')
                writer.writerow(parts)
                print(f"✅ {line}")
    except KeyboardInterrupt:
        print("\n🛑 Logging stopped by user.")

# === Close serial connection ===
ser.close()
