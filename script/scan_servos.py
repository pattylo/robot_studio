#!/usr/bin/env python3
"""Scan for connected LX16A servos."""

import time
import serial

def scan_servos_simple(port="/dev/ttyUSB0", max_id=20):
    """Scan for servos by sending direct commands."""
    print(f"Scanning for servos on {port}...")
    print(f"Testing IDs 0-{max_id} with direct serial commands...\n")
    
    ser = serial.Serial(port, baudrate=115200, timeout=0.1)
    time.sleep(0.1)
    
    found_servos = []
    
    for servo_id in range(max_id + 1):
        print(f"Testing ID {servo_id}... ", end='', flush=True)
        
        # Send a read position command (cmd 28)
        length = 3
        packet = [0x55, 0x55, servo_id, length, 28]
        checksum = (~sum(packet[2:]) & 0xFF)
        packet.append(checksum)
        
        ser.reset_input_buffer()
        ser.write(bytes(packet))
        time.sleep(0.05)
        
        # Try to read response
        response = ser.read(10)
        
        if len(response) >= 7:
            print(f"✓ FOUND! (response: {response.hex()})")
            found_servos.append(servo_id)
        else:
            print("✗")
        
        time.sleep(0.05)
    
    ser.close()
    
    print("\n" + "="*50)
    if found_servos:
        print(f"Found {len(found_servos)} servo(s): {found_servos}")
    else:
        print("No servos found!")
        print("\nTroubleshooting:")
        print("1. Check servo power supply")
        print("2. Check serial connection (TX/RX/GND)")
        print("3. Verify servos are in servo mode, not motor mode")
    print("="*50)

if __name__ == "__main__":
    scan_servos_simple()